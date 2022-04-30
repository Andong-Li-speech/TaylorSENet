import torch
from torch import Tensor
import torch.nn as nn
import math
from utils.utils import NormSwitch

class TaylorSENet(nn.Module):
    def __init__(self,
                 cin: int = 2,
                 k1: list = [1, 3],
                 k2: list = [2, 3],
                 c: int = 64,
                 kd1: int = 5,
                 cd1: int = 64,
                 d_feat: int = 256,
                 dilations: list = [1, 2, 5, 9],
                 p: int = 2,
                 fft_num: int = 320,
                 order_num: int = 3,
                 intra_connect: str = "cat",
                 inter_connect: str = "add",
                 norm_type: str = "IN",
                 is_causal: bool = True,
                 is_u2: bool = True,
                 is_param_share: bool = False,
                 is_encoder_share: bool = False,
                 ):
        super(TaylorSENet, self).__init__()
        self.cin = cin
        self.k1 = tuple(k1)
        self.k2 = tuple(k2)
        self.c = c
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.dilations = dilations
        self.p = p
        self.fft_num = fft_num
        self.order_num = order_num
        self.intra_connect = intra_connect
        self.inter_connect = inter_connect
        self.norm_type = norm_type
        self.is_causal = is_causal
        self.is_u2 = is_u2
        self.is_param_share = is_param_share
        self.is_encoder_share = is_encoder_share

        self.zeroorderblock = ZeroOrderBlock(cin//2, tuple(k1), tuple(k2), c, kd1, cd1, d_feat, dilations, p, intra_connect,
                                             inter_connect, norm_type,
                                             is_causal, is_u2)
        if not is_encoder_share:
            if is_u2:
                self.separate_en = U2Net_Encoder(cin, tuple(k1), tuple(k2), c, intra_connect, norm_type)
            else:
                self.separate_en = UNet_Encoder(cin, tuple(k1), c, norm_type)

        if is_param_share:
            highorderblock_list = []
            highorderblock_list.append(HighOrderBlock(kd1, cd1, d_feat, dilations, p, fft_num, is_causal, norm_type))
        else:
            highorderblock_list = []
            for i in range(order_num):
                highorderblock_list.append(HighOrderBlock(kd1, cd1, d_feat, dilations, p, fft_num, is_causal, norm_type))
        self.highorderblock_list = nn.ModuleList(highorderblock_list)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        :param inputs: (B,2,T,F)
        :return: (B,2,T,F)
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        inputs_mag, inputs_phase = torch.norm(inputs, dim=1), torch.atan2(inputs[:, -1, ...], inputs[:, 0, ...])
        zeroorder_gain, feature_head = self.zeroorderblock(inputs_mag)
        zeroorder_mag = zeroorder_gain * inputs_mag
        zero_term = torch.stack((zeroorder_mag*torch.cos(inputs_phase), zeroorder_mag*torch.sin(inputs_phase)), dim=1)

        if not self.is_encoder_share:
            feature_head, _ = self.separate_en(inputs)
            b_size, c, seq_len, freq_num = feature_head.shape
            feature_head = feature_head.transpose(-2, -1).contiguous()
            feature_head = feature_head.view(b_size, -1, seq_len)

        out_term, pre_term = zero_term, zero_term
        for order_id in range(self.order_num):
            if self.is_param_share:
                update_term = self.highorderblock_list[0](feature_head, pre_term) + order_id * pre_term
                pre_term = update_term
                out_term = out_term + update_term / math.factorial(order_id+1)
            else:
                update_term = self.highorderblock_list[order_id](feature_head, pre_term) + order_id * pre_term
                pre_term = update_term
                out_term = out_term + update_term / math.factorial(order_id+1)
        return out_term


class ZeroOrderBlock(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 k2: tuple,
                 c: int,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 dilations: list,
                 p: int,
                 intra_connect: str,
                 inter_connect: str,
                 norm_type: str,
                 is_causal: bool,
                 is_u2: bool,
                 ):
        super(ZeroOrderBlock, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.dilations = dilations
        self.p = p
        self.intra_connect = intra_connect
        self.inter_connect = inter_connect
        self.norm_type = norm_type
        self.is_causal = is_causal
        self.is_u2 = is_u2

        if self.is_u2:
            self.en = U2Net_Encoder(cin, k1, k2, c, intra_connect, norm_type)
            self.de = U2Net_Decoder(c, k1, k2, intra_connect, inter_connect, norm_type)
        else:
            self.en = UNet_Encoder(cin, k1, c, norm_type)
            self.de = UNet_Decoder(c, k1, inter_connect, norm_type)

        tcm_list = []
        for i in range(p):
            tcm_list.append(TCMList(kd1, cd1, d_feat, dilations, is_causal, norm_type))
        self.tcms = nn.ModuleList(tcm_list)

    def forward(self, inputs: Tensor) -> tuple:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        en_x, en_list = self.en(inputs)
        b_size, c, seq_len, freq_num = en_x.shape
        x = en_x.transpose(-2, -1).contiguous()
        feature_head = x.view(b_size, c*freq_num, seq_len)
        tcm_x = feature_head
        for i in range(self.p):
            tcm_x = self.tcms[i](tcm_x)

        x = tcm_x.view(b_size, c, freq_num, seq_len)
        x = x.transpose(-2, -1).contiguous()
        gain = self.de(x, en_list)
        return gain, feature_head


class HighOrderBlock(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 dilations: list,
                 p: int,
                 fft_num: int,
                 is_causal: bool,
                 norm_type: str,
                 ):
        super(HighOrderBlock, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.dilations = dilations
        self.p = p
        self.fft_num = fft_num
        self.is_causal = is_causal
        self.norm_type = norm_type

        in_feat = (fft_num//2+1)*2 + d_feat
        self.in_conv = nn.Conv1d(in_feat, d_feat, 1)

        tcm_list = []
        for i in range(p):
            tcm_list.append(TCMList(kd1, cd1, d_feat, dilations, is_causal, norm_type))
        self.tcms = nn.ModuleList(tcm_list)
        self.real_resi, self.imag_resi = nn.Conv1d(d_feat, fft_num//2+1, 1), nn.Conv1d(d_feat, fft_num//2+1, 1)

    def forward(self, en_x: Tensor, pre_x: Tensor) -> Tensor:
        """
        :param en_x:  (B, C, T)
        :param pre_x: (B, 2, T, F)
        :return:  (B, 2, T, F)
        """
        #assert en_x.dim() == 3 and pre_x.dim() == 4, 'dimension is mismatched'
        # fuse the features
        b_size, _, seq_len, freq_num = pre_x.shape
        x1 = pre_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        x = torch.cat((en_x, x1), dim=1)
        # in conv
        x = self.in_conv(x)
        # STCMs
        for i in range(self.p):
            x = self.tcms[i](x)
        # generate real and imaginary parts
        xr, xi = self.real_resi(x).transpose(-2, -1).contiguous(), self.imag_resi(x).transpose(-2, -1).contiguous()
        return torch.stack((xr, xi), dim=1)


class UNet_Encoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 c: int,
                 norm_type: str,
                 ):
        super(UNet_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.c = c
        self.norm_type = norm_type
        kernel_begin = (1, 5)
        stride = (1, 2)
        c_final = 64
        unet = []
        unet.append(nn.Sequential(
            GateConv2d(cin, c, kernel_begin, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c_final, k1, (1,2), padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c_final),
            nn.PReLU(c_final)))
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor) -> tuple:
        en_list = []
        for i in range(len(self.unet_list)):
            x = self.unet_list[i](x)
            en_list.append(x)
        return x, en_list


class UNet_Decoder(nn.Module):
    def __init__(self,
                 c: int,
                 k1: tuple,
                 inter_connect: str,
                 norm_type: str,
                 ):
        super(UNet_Decoder, self).__init__()
        self.k1 = k1
        self.c = c
        self.inter_connect = inter_connect
        self.norm_type = norm_type
        c_begin = 64
        c_end = 16
        kernel_end = (1, 5)
        stride = (1, 2)
        unet = []
        if inter_connect == "add":
            factor = 1
        elif inter_connect == "cat":
            factor = 2
        else:
            raise Exception("only add and cat are supported")
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c_begin*factor, c, k1, stride),
                NormSwitch(norm_type, "2D", c),
                nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConvTranspose2d(c*factor, c, k1, stride),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)
        ))
        unet.append(nn.Sequential(
            GateConvTranspose2d(c*factor, c, k1, stride),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)
        ))
        unet.append(nn.Sequential(
            GateConvTranspose2d(c*factor, c, k1, stride),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)
        ))
        unet.append(nn.Sequential(
            GateConvTranspose2d(c*factor, c_end, kernel_end, stride),
            NormSwitch(norm_type, "2D", c_end),
            nn.PReLU(c_end),
            nn.Conv2d(c_end, 1, (1, 1), (1, 1)),
            nn.Sigmoid()))
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor, en_list: list) -> Tensor:
        if self.inter_connect == "cat":
            for i in range(len(self.unet_list)):
                tmp = torch.cat((x, en_list[-(i+1)]), dim=1)
                x = self.unet_list[i](tmp)
        elif self.inter_connect == "add":
            for i in range(len(self.unet_list)):
                tmp = x + en_list[-(i+1)]
                x = self.unet_list[i](tmp)
        return x.squeeze(dim=1)


class U2Net_Encoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 k2: tuple,
                 c: int,
                 intra_connect: str,
                 norm_type: str,
                 ):
        super(U2Net_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        c_last = 64
        kernel_begin = (1, 5)
        stride = (1, 2)
        meta_unet = []
        meta_unet.append(
            En_unet_module(cin, c, kernel_begin, k2, intra_connect, norm_type, scale=4, de_flag=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=3, de_flag=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=2, de_flag=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=1, de_flag=False))
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConv2d(c, c_last, k1, stride, (0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c_last),
            nn.PReLU(c_last)
        )

    def forward(self, x: Tensor) -> tuple:
        en_list = []
        for i in range(len(self.meta_unet_list)):
            x = self.meta_unet_list[i](x)
            en_list.append(x)
        x = self.last_conv(x)
        en_list.append(x)
        return x, en_list


class U2Net_Decoder(nn.Module):
    def __init__(self,
                 c: int,
                 k1: tuple,
                 k2: tuple,
                 intra_connect: str,
                 inter_connect: str,
                 norm_type: str,
                 ):
        super(U2Net_Decoder, self).__init__()
        self.c = c
        self.k1 = k1
        self.k2 = k2
        self.intra_connect = intra_connect
        self.inter_connect = inter_connect
        self.norm_type = norm_type
        c_begin = 64
        c_end = 16
        kernel_end = (1, 5)
        stride = (1, 2)
        meta_unet = []
        if inter_connect == "add":
            factor = 1
        elif inter_connect == "cat":
            factor = 2
        else:
            raise Exception("only add and cat are supported")
        meta_unet.append(
            En_unet_module(c_begin*factor, c, k1, k2, intra_connect, norm_type, scale=1, de_flag=True)
        )
        meta_unet.append(
            En_unet_module(c*factor, c, k1, k2, intra_connect, norm_type, scale=2, de_flag=True)
        )
        meta_unet.append(
            En_unet_module(c*factor, c, k1, k2, intra_connect, norm_type, scale=3, de_flag=True)
        )
        meta_unet.append(
            En_unet_module(c*factor, c, k1, k2, intra_connect, norm_type, scale=4, de_flag=True)
        )
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConvTranspose2d(c*factor, c_end, kernel_end, stride),
            NormSwitch(norm_type, "2D", c_end),
            nn.PReLU(c_end),
            nn.Conv2d(c_end, 1, (1, 1), (1, 1)),
            nn.Sigmoid())

    def forward(self, x: Tensor, en_list: list) -> Tensor:
        if self.inter_connect == "add":
            for i in range(len(self.meta_unet_list)):
                tmp = x + en_list[-(i+1)]
                x = self.meta_unet_list[i](tmp)
            x = x + en_list[0]
            x = self.last_conv(x)
        elif self.inter_connect == "cat":
            for i in range(len(self.meta_unet_list)):
                tmp = torch.cat((x, en_list[-(i+1)]), dim=1)
                x = self.meta_unet_list[i](tmp)
            x = torch.cat((x, en_list[0]), dim=1)
            x = self.last_conv(x)
        return x.squeeze(dim=1)


class En_unet_module(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k1: tuple,
                 k2: tuple,
                 intra_connect: str,
                 norm_type: str,
                 scale: int,
                 de_flag: bool = False):
        super(En_unet_module, self).__init__()
        self.cin = cin
        self.cout = cout
        self.k1 = k1
        self.k2 = k2
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.scale = scale
        self.de_flag = de_flag

        in_conv_list = []
        if de_flag is False:
            in_conv_list.append(GateConv2d(cin, cout, k1, (1, 2), (0, 0, k1[0]-1, 0)))
        else:
            in_conv_list.append(GateConvTranspose2d(cin, cout, k1, (1, 2)))
        in_conv_list.append(NormSwitch(norm_type, "2D", cout))
        in_conv_list.append(nn.PReLU(cout))
        self.in_conv = nn.Sequential(*in_conv_list)

        enco_list, deco_list = [], []
        for _ in range(scale):
            enco_list.append(Conv2dunit(k2, cout, norm_type))
        for i in range(scale):
            if i == 0:
                deco_list.append(Deconv2dunit(k2, cout, "add", norm_type))
            else:
                deco_list.append(Deconv2dunit(k2, cout, intra_connect, norm_type))
        self.enco = nn.ModuleList(enco_list)
        self.deco = nn.ModuleList(deco_list)
        self.skip_connect = Skip_connect(intra_connect)

    def forward(self, inputs: Tensor) -> Tensor:
        x_resi = self.in_conv(inputs)
        x = x_resi
        x_list = []
        for i in range(len(self.enco)):
            x = self.enco[i](x)
            x_list.append(x)

        for i in range(len(self.deco)):
            if i == 0:
                x = self.deco[i](x)
            else:
                x_con = self.skip_connect(x, x_list[-(i+1)])
                x = self.deco[i](x_con)
        x_resi = x_resi + x
        del x_list
        return x_resi

class Conv2dunit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 norm_type: str,
                 ):
        super(Conv2dunit, self).__init__()
        self.k, self.c = k, c
        self.norm_type = norm_type
        k_t = k[0]
        stride = (1, 2)
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConstantPad2d((0, 0, k_t-1, 0), value=0.),
                nn.Conv2d(c, c, k, stride),
                NormSwitch(norm_type, "2D", c),
                nn.PReLU(c)
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(c, c, k, stride),
                NormSwitch(norm_type, "2D", c),
                nn.PReLU(c)
            )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class Deconv2dunit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 intra_connect: str,
                 norm_type: str,
                 ):
        super(Deconv2dunit, self).__init__()
        self.k, self.c = k, c
        self.intra_connect = intra_connect
        k_t = k[0]
        stride = (1,2)
        deconv_list = []
        if self.intra_connect == "add":
            if k_t > 1:
                deconv_list.append(nn.ConvTranspose2d(c, c, k, stride)),
                deconv_list.append(Chomp_T(k_t-1))
            else:
                deconv_list.append(nn.ConvTranspose2d(c, c, k, stride))
        elif self.intra_connect == "cat":
            if k_t > 1:
                deconv_list.append(nn.ConvTranspose2d(2*c, c, k, stride))
                deconv_list.append(Chomp_T(k_t-1))
            else:
                deconv_list.append(nn.ConvTranspose2d(2*c, c, k, stride))
        deconv_list.append(NormSwitch(norm_type, "2D", c))
        deconv_list.append(nn.PReLU(c))
        self.deconv = nn.Sequential(*deconv_list)

    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.dim() == 4
        return self.deconv(inputs)


class GateConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 padding: tuple,):
        super(GateConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding, value=0.),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size, stride=stride))
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                  stride=stride)
    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class GateConvTranspose2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 ):
        super(GateConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                   stride=stride),
                Chomp_T(k_t-1))
        else:
            self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                           stride=stride)

    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.dim() == 4
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class Skip_connect(nn.Module):
    def __init__(self, connect):
        super(Skip_connect, self).__init__()
        self.connect = connect

    def forward(self, x_main, x_aux):
        if self.connect == "add":
            x = x_main + x_aux
        elif self.connect == "cat":
            x = torch.cat((x_main, x_aux), dim=1)
        return x


class TCMList(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 dilations: list,
                 is_causal: bool,
                 norm_type : str,
                 ):
        super(TCMList, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.dilations = dilations
        self.is_causal = is_causal
        self.norm_type = norm_type
        tcm_list = []
        for i in range(len(dilations)):
            tcm_list.append(SqueezedTCM(kd1, cd1, dilation=dilations[i], d_feat=d_feat, is_causal=is_causal,
                                        norm_type=norm_type))
        self.tcm_list = nn.ModuleList(tcm_list)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for i in range(len(self.dilations)):
            x = self.tcm_list[i](x)
        return x


class SqueezedTCM(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 dilation: int,
                 d_feat: int,
                 is_causal: bool,
                 norm_type: str,
                 ):
        super(SqueezedTCM, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.dilation = dilation
        self.d_feat = d_feat
        self.is_causal = is_causal
        self.norm_type = norm_type

        self.in_conv = nn.Conv1d(d_feat, cd1, kernel_size=1, bias=False)
        if is_causal:
            pad = ((kd1-1)*dilation, 0)
        else:
            pad = ((kd1-1)*dilation//2, (kd1-1)*dilation//2)
        self.left_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.ConstantPad1d(pad, value=0.),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False)
        )
        self.right_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.ConstantPad1d(pad, value=0.),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.Conv1d(cd1, d_feat, kernel_size=1, bias=False)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        resi = inputs
        x = self.in_conv(inputs)
        x = self.left_conv(x) * self.right_conv(x)
        x = self.out_conv(x)
        x = x + resi
        return x

class Chomp_T(nn.Module):
    def __init__(self, t):
        super(Chomp_T, self).__init__()
        self.t = t

    def forward(self, x):
        return x[:, :, :-self.t, :]

if __name__ == '__main__':
    net = TaylorSENet(cin=2,
                      k1=[1, 3],
                      k2=[2, 3],
                      c=64,
                      kd1=5,
                      cd1=64,
                      d_feat=256,
                      dilations=[1, 2, 5, 9],
                      p=2,
                      fft_num=320,
                      order_num=3,
                      intra_connect="cat",
                      inter_connect="cat",
                      norm_type="IN",
                      is_causal=True,
                      is_u2=True,
                      is_param_share=False,
                      is_encoder_share=False).cuda()
    from utils.utils import numParams
    print(f"The number of trainable parameters:{numParams(net)}")
    x = torch.rand([4,2,101,161]).cuda()
    y = net(x)
    print('{}->{}'.format(x.shape, y.shape))
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(net, (2, 101, 161))
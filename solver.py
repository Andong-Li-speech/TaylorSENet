import numpy as np
import os
import torch
import torch.nn as nn
import time
import importlib
from utils.utils import logger_print
import hdf5storage
from utils.utils import cal_pesq
from utils.utils import cal_stoi
from utils.utils import cal_sisnr
train_epoch, val_epoch, val_metric_epoch = [], [], []  # for loss, loss and metric score
# from torch.utils.tensorboard import SummaryWriter


class Solver(object):
    def __init__(self,
                 data,
                 net,
                 optimizer,
                 save_name_dict,
                 args,
                 ):
        self.train_dataloader = data["train_loader"]
        self.val_dataloader = data["val_loader"]
        self.net = net
        # optimizer part
        self.optimizer = optimizer
        self.lr = args["optimizer"]["lr"]
        self.gradient_norm = args["optimizer"]["gradient_norm"]
        self.epochs = args["optimizer"]["epochs"]
        self.halve_lr = args["optimizer"]["halve_lr"]
        self.early_stop = args["optimizer"]["early_stop"]
        self.halve_freq = args["optimizer"]["halve_freq"]
        self.early_stop_freq = args["optimizer"]["early_stop_freq"]
        self.print_freq = args["optimizer"]["print_freq"]
        self.metric_options = args["optimizer"]["metric_options"]
        # loss part
        self.loss_path = args["loss_function"]["path"]
        self.commag_loss = args["loss_function"]["stagewise"]["classname"]
        self.alpha = args["loss_function"]["alpha"]
        self.l_type = args["loss_function"]["l_type"]
        # signal part
        self.sr = args["signal"]["sr"]
        self.win_size = args["signal"]["win_size"]
        self.win_shift = args["signal"]["win_shift"]
        self.fft_num = args["signal"]["fft_num"]
        self.is_compress = args["signal"]["is_compress"]
        # path part
        self.is_checkpoint = args["path"]["is_checkpoint"]
        self.is_resume_reload = args["path"]["is_resume_reload"]
        self.checkpoint_load_path = args["path"]["checkpoint_load_path"]
        self.checkpoint_load_filename = args["path"]["checkpoint_load_filename"]
        self.loss_save_path = args["path"]["loss_save_path"]
        self.model_best_path = args["path"]["model_best_path"]
        # sava name
        self.loss_save_filename = save_name_dict["loss_filename"]
        self.best_model_save_filename = save_name_dict["best_model_filename"]
        self.checkpoint_save_filename = save_name_dict["checkpoint_filename"]

        self.train_loss = torch.Tensor(self.epochs)
        self.val_loss = torch.Tensor(self.epochs)
        # set loss funcs
        loss_module = importlib.import_module(self.loss_path)
        self.commag_loss = getattr(loss_module, self.commag_loss)(self.alpha, self.l_type)
        self._reset()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # summarywriter
        # self.tensorboard_path = "./" + args["path"]["logging_path"] + "/" + args["save"]["tensorboard_filename"]
        # if not os.path.exists(self.tensorboard_path):
        #     os.makedirs(self.tensorboard_path)
        #self.writer = SummaryWriter(self.tensorboard_path, max_queue=5, flush_secs=30)

    def _reset(self):
        # Reset
        if self.is_resume_reload:
            checkpoint = torch.load(os.path.join(self.checkpoint_load_path, self.checkpoint_load_filename))
            self.net.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["start_epoch"]
            self.prev_val_loss = checkpoint["val_loss"]  # val loss
            self.prev_val_metric = checkpoint["val_metric"]
            self.best_val_metric = checkpoint["best_val_metric"]
            self.val_no_impv = checkpoint["val_no_impv"]
            self.halving = checkpoint["halving"]
        else:
            self.start_epoch = 0
            self.prev_val_loss = float("inf")
            self.prev_val_metric = -float("inf")
            self.best_val_metric = -float("inf")
            self.val_no_impv = 0
            self.halving = False

    def train(self):
        logger_print("Begin to train....")
        self.net.to(self.device)
        for epoch in range(self.start_epoch, self.epochs):
            begin_time = time.time()
            # training phase
            logger_print("-" * 90)
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger_print(f"Epoch id:{int(epoch + 1)}, Training phase, Start time:{start_time}")
            self.net.train()
            train_avg_loss = self._run_one_epoch(epoch, val_opt=False)
            # self.writer.add_scalar(f"Loss/Training_Loss", train_avg_loss, epoch)
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger_print(f"Epoch if:{int(epoch + 1)}, Training phase, End time:{end_time}, "
                         f"Training loss:{train_avg_loss}")

            # Cross val
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger_print(f"Epoch id:{int(epoch + 1)}, Validation phase, Start time:{start_time}")
            self.net.eval()    # norm and dropout is off
            val_avg_loss, val_avg_metric = self._run_one_epoch(epoch, val_opt=True)
            # self.writer.add_scalar(f"Loss/Validation_Loss", val_avg_loss, epoch)
            # self.writer.add_scalar(f"Loss/Validation_Metric", val_avg_metric, epoch)
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger_print(f"Epoch if:{int(epoch + 1)}, Validation phase, End time:{end_time}, "
                         f"Validation loss:{val_avg_loss}, Validation metric score:{val_avg_metric}")
            end_time = time.time()
            print(f"{end_time-begin_time}s in {epoch+1}th epoch")
            logger_print("-" * 90)

            # whether to save checkpoint at current epoch
            if self.is_checkpoint:
                cpk_dic = {}
                cpk_dic["model_state_dict"] = self.net.state_dict()
                cpk_dic["optimizer_state_dict"] = self.optimizer.state_dict()
                cpk_dic["train_loss"] = train_avg_loss
                cpk_dic["val_loss"] = val_avg_loss
                cpk_dic["val_metric"] = val_avg_metric
                cpk_dic["best_val_metric"] = self.best_val_metric
                cpk_dic["start_epoch"] = epoch+1
                cpk_dic["val_no_impv"] = self.val_no_impv
                cpk_dic["halving"] = self.halving
                torch.save(cpk_dic, os.path.join(self.checkpoint_load_path, "Epoch_{}_{}_{}".format(epoch+1,
                                                        self.net.__class__.__name__, self.checkpoint_save_filename)))
            # record loss
            # self.train_loss[epoch] = train_avg_loss
            # self.val_loss[epoch] = val_avg_loss

            train_epoch.append(train_avg_loss)
            val_epoch.append(val_avg_loss)
            val_metric_epoch.append(val_avg_metric)

            # save loss
            loss = {}
            loss["train_loss"] = train_epoch
            loss["val_loss"] = val_epoch
            loss["val_metric"] = val_metric_epoch

            if not self.is_resume_reload:
                hdf5storage.savemat(os.path.join(self.loss_save_path, self.loss_save_filename), loss)
            else:
                hdf5storage.savemat(os.path.join(self.loss_save_path, "resume_cpk_{}".format(self.loss_save_filename)),
                                    loss)

            # lr halve and Early stop
            if self.halve_lr:
                if val_avg_metric <= self.prev_val_metric:
                    self.val_no_impv += 1
                    if self.val_no_impv == self.halve_freq:
                        self.halving = True
                    if (self.val_no_impv >= self.early_stop_freq) and self.early_stop:
                        logger_print("No improvements and apply early-stopping")
                        break
                else:
                    self.val_no_impv = 0

            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state["param_groups"][0]["lr"] = optim_state["param_groups"][0]["lr"] / 2.0
                self.optimizer.load_state_dict(optim_state)
                logger_print("Learning rate is adjusted to %5f" % (optim_state["param_groups"][0]["lr"]))
                self.halving = False
            self.prev_val_metric = val_avg_metric

            if val_avg_metric > self.best_val_metric:
                self.best_val_metric = val_avg_metric
                torch.save(self.net.state_dict(), os.path.join(self.model_best_path, self.best_model_save_filename))
                logger_print(f"Find better model, saving to {self.best_model_save_filename}")
            else:
                logger_print("Did not find better model")

    @torch.no_grad()
    def _val_batch(self, batch_info):
        batch_mix_wav = batch_info.feats.to(self.device)  # (B,L)
        batch_target_wav = batch_info.labels.to(self.device)  # (B,L)
        batch_wav_len_list = batch_info.frame_mask_list
        real_len = batch_mix_wav.shape[-1]

        # stft
        b_size, wav_len = batch_mix_wav.shape
        win_size, win_shift = int(self.sr*self.win_size), int(self.sr*self.win_shift)
        batch_mix_stft = torch.stft(
            batch_mix_wav,
            n_fft=self.fft_num,
            hop_length=win_shift,
            win_length=win_size,
            window=torch.hann_window(win_size).to(self.device))   # (B,F,T,2)
        batch_target_stft = torch.stft(
            batch_target_wav,
            n_fft=self.fft_num,
            hop_length=win_shift,
            win_length=win_size,
            window=torch.hann_window(win_size).to(self.device))  # (B,F,T,2)

        batch_frame_list = []
        for i in range(len(batch_wav_len_list)):
            curr_frame_num = (batch_wav_len_list[i]-win_size+win_size)//win_shift+1  # center case
            batch_frame_list.append(curr_frame_num)
        _, freq_num, seq_len, _ = batch_mix_stft.shape
        if self.is_compress:
            # mix
            batch_mix_mag, batch_mix_phase = torch.norm(batch_mix_stft, dim=-1)**0.5, \
                                             torch.atan2(batch_mix_stft[..., -1], batch_mix_stft[..., 0])
            batch_mix_stft = torch.stack((batch_mix_mag*torch.cos(batch_mix_phase),
                                          batch_mix_mag*torch.sin(batch_mix_phase)), dim=-1)
            # target
            batch_target_mag, batch_target_phase = torch.norm(batch_target_stft, dim=-1)**0.5, \
                                                   torch.atan2(batch_target_stft[...,-1],
                                                               batch_target_stft[...,0])
            batch_target_stft = torch.stack((batch_target_mag*torch.cos(batch_target_phase),
                                             batch_target_mag*torch.sin(batch_target_phase)), dim=-1)

        # convert, mix/target: (B,2,T,F)
        batch_mix_stft = batch_mix_stft.permute(0,3,2,1)  # (B,2,T,F)
        batch_target_stft = batch_target_stft.permute(0,3,2,1)  # (B,2,T,F)

        # net predict
        batch_est = self.net(batch_mix_stft)  # (B,2,T,F)
        # cal loss
        batch_loss = self.commag_loss(batch_est, batch_target_stft, batch_frame_list)
        # cal metric loss, (B,F,T,2)
        batch_mix_stft = batch_mix_stft.permute(0,3,2,1)  # (B,F,T,2)
        batch_est_stft = batch_est.permute(0,3,2,1)  # (B,F,T,2)
        batch_target_stft = batch_target_stft.permute(0,2,3,1)  # (B,F,T,2)
        if self.is_compress:
            # mix
            batch_mix_mag, batch_mix_phase = torch.norm(batch_mix_stft, dim=-1)**2.0, \
                                             torch.atan2(batch_mix_stft[..., -1], batch_mix_stft[...,0])
            batch_mix_stft = torch.stack((batch_mix_mag*torch.cos(batch_mix_phase),
                                          batch_mix_mag*torch.sin(batch_mix_phase)), dim=-1)
            # est
            batch_spec_mag, batch_spec_phase = torch.norm(batch_est_stft, dim=-1)**2.0,\
                                               torch.atan2(batch_est_stft[..., -1], batch_est_stft[...,0])
            batch_est_stft = torch.stack((batch_spec_mag*torch.cos(batch_spec_phase),
                                          batch_spec_mag*torch.sin(batch_spec_phase)), dim=-1)
            # target
            batch_target_mag, batch_target_phase = torch.norm(batch_target_stft, dim=-1)**2.0, \
                                               torch.atan2(batch_target_stft[...,-1], batch_target_stft[...,0])
            batch_target_stft = torch.stack((batch_target_mag*torch.cos(batch_target_phase),
                                             batch_target_mag*torch.sin(batch_target_phase)), dim=-1)
        batch_mix_wav = torch.istft(batch_mix_stft,
                                    n_fft=self.fft_num,
                                    hop_length=win_shift,
                                    win_length=win_size,
                                    window=torch.hann_window(win_size).to(self.device),
                                    length=real_len
                                    )  # (B,L)
        batch_est_wav = torch.istft(batch_est_stft,
                                    n_fft=self.fft_num,
                                    hop_length=win_shift,
                                    win_length=win_size,
                                    window=torch.hann_window(win_size).to(self.device),
                                    length=real_len
                                    )  # (B,L)
        batch_target_wav = torch.istft(batch_target_stft,
                                       n_fft=self.fft_num,
                                       hop_length=win_shift,
                                       win_length=win_size,
                                       window=torch.hann_window(win_size).to(self.device),
                                       length=real_len
                                       )  # (B,L)
        loss_dict = {}
        loss_dict["mse_loss"] = batch_loss.item()
        # create mask
        mask_list = []
        for id in range(b_size):
            mask_list.append(torch.ones((batch_wav_len_list[id])))
        wav_mask = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True).to(batch_mix_stft.device)  # (B,L)
        batch_mix_wav, batch_target_wav, batch_est_wav = (batch_mix_wav*wav_mask).cpu().numpy(), \
                                                         (batch_target_wav*wav_mask).cpu().numpy(), \
                                                         (batch_est_wav*wav_mask).cpu().numpy()
        if "SISNR" in self.metric_options:
            unpro_score_list, pro_score_list = [], []
            for id in range(batch_mix_wav.shape[0]):
                unpro_score_list.append(cal_sisnr(id, batch_mix_wav, batch_target_wav, self.sr))
                pro_score_list.append(cal_sisnr(id, batch_est_wav, batch_target_wav, self.sr))
            unpro_score_list, pro_score_list = np.asarray(unpro_score_list), np.asarray(pro_score_list)
            unpro_sisnr_mean_score, pro_sisnr_mean_score = np.mean(unpro_score_list), np.mean(pro_score_list)
            loss_dict["unpro_metric"] = unpro_sisnr_mean_score
            loss_dict["pro_metric"] = pro_sisnr_mean_score
        if "NB-PESQ" in self.metric_options:
            unpro_score_list, pro_score_list = [], []
            for id in range(batch_mix_wav.shape[0]):
                unpro_score_list.append(cal_pesq(id, batch_mix_wav, batch_target_wav, self.sr))
                pro_score_list.append(cal_pesq(id, batch_est_wav, batch_target_wav, self.sr))
            unpro_score_list, pro_score_list = np.asarray(unpro_score_list), \
                                               np.asarray(pro_score_list)
            unpro_pesq_mean_score, pro_pesq_mean_score = np.mean(unpro_score_list), np.mean(pro_score_list)
            loss_dict["unpro_metric"] = unpro_pesq_mean_score
            loss_dict["pro_metric"] = pro_pesq_mean_score
        if "ESTOI" in self.metric_options:
            unpro_score_list, pro_score_list = [], []
            for id in range(batch_mix_wav.shape[0]):
                unpro_score_list.append(cal_stoi(id, batch_mix_wav, batch_target_wav, self.sr))
                pro_score_list.append(cal_stoi(id, batch_mix_wav, batch_target_wav, self.sr))
            unpro_score_list, pro_score_list = np.asarray(unpro_score_list), \
                                               np.asarray(pro_score_list)
            unpro_estoi_mean_score, pro_estoi_mean_score = np.mean(unpro_score_list), np.mean(pro_score_list)
            loss_dict["unpro_metric"] = unpro_estoi_mean_score
            loss_dict["pro_metric"] = pro_estoi_mean_score
        return loss_dict

    def _train_batch(self, batch_info):
        batch_mix_wav = batch_info.feats.to(self.device)  # (B,L)
        batch_target_wav = batch_info.labels.to(self.device)  # (B,L)
        batch_wav_len_list = batch_info.frame_mask_list

        # stft
        win_size, win_shift = int(self.sr*self.win_size), int(self.sr*self.win_shift)
        batch_mix_stft = torch.stft(
            batch_mix_wav,
            n_fft=self.fft_num,
            hop_length=win_shift,
            win_length=win_size,
            window=torch.hann_window(win_size).to(batch_mix_wav.device))  # (B,F,T,2)
        batch_target_stft = torch.stft(
            batch_target_wav,
            n_fft=self.fft_num,
            hop_length=win_shift,
            win_length=win_size,
            window=torch.hann_window(win_size).to(batch_target_wav.device))  # (B,F,T,2)
        batch_frame_list = []
        for i in range(len(batch_wav_len_list)):
            curr_frame_num = (batch_wav_len_list[i] - win_size + win_size) // win_shift + 1
            batch_frame_list.append(curr_frame_num)

        if self.is_compress:  # here only apply to target and bf as feat-compression has been applied within the network
            # mix
            batch_mix_mag, batch_mix_phase = torch.norm(batch_mix_stft, dim=-1)**0.5, \
                                             torch.atan2(batch_mix_stft[..., -1], batch_mix_stft[..., 0])
            batch_mix_stft = torch.stack((batch_mix_mag*torch.cos(batch_mix_phase),
                                          batch_mix_mag*torch.sin(batch_mix_phase)), dim=-1)
            # target
            batch_target_mag, batch_target_phase = torch.norm(batch_target_stft, dim=-1)**0.5, \
                                                   torch.atan2(batch_target_stft[..., -1],
                                                               batch_target_stft[..., 0])
            batch_target_stft = torch.stack((batch_target_mag*torch.cos(batch_target_phase),
                                             batch_target_mag*torch.sin(batch_target_phase)), dim=-1)

        # convert, (B,2,T,F)
        batch_mix_stft = batch_mix_stft.permute(0,3,2,1)  # (B,2,T,F)
        batch_target_stft = batch_target_stft.permute(0,3,2,1)  # (B,2,T,F)

        with torch.enable_grad():
            batch_est_list = self.net(batch_mix_stft)  # (B,2,T,F)

        # stagewise loss
        batch_loss = self.commag_loss(batch_est_list, batch_target_stft, batch_frame_list)
        # params update
        self.update_params(batch_loss)
        loss_dict = {}
        loss_dict["mse_loss"] = batch_loss.item()
        return loss_dict

    def _run_one_epoch(self, epoch, val_opt=False):
        # training phase
        if not val_opt:
            data_loader = self.train_dataloader
            total_loss = 0.
            start_time = time.time()
            for batch_id, batch_info in enumerate(data_loader.get_data_loader()):
                loss_dict = self._train_batch(batch_info)
                total_loss += loss_dict["mse_loss"]
                if batch_id % self.print_freq == 0:
                    logger_print(
                        "Epoch:{:d}, Iter:{:d}, Average loss:{:.4f}, Time: {:d}ms/batch".
                            format(epoch+1, int(batch_id), total_loss/(batch_id+1),
                                                           int(1000*(time.time()-start_time)/(batch_id+1))))
            return total_loss / (batch_id+1)

        else:  # validation phase
            data_loder = self.val_dataloader
            total_sp_loss, total_pro_metric_loss, total_unpro_metric_loss = 0., 0., 0.
            start_time = time.time()
            for batch_id, batch_info in enumerate(data_loder.get_data_loader()):
                loss_dict = self._val_batch(batch_info)
                assert len(self.metric_options) == 1, "only one metric is supported to output in the val phase"
                total_sp_loss += loss_dict["mse_loss"]
                total_unpro_metric_loss += loss_dict["unpro_metric"]
                total_pro_metric_loss += loss_dict["pro_metric"]
                if batch_id % self.print_freq == 0:
                    logger_print(
                        "Epoch:{:d}, Iter:{:d}, Average spectral loss:{:.4f}, Average unpro metric score:{:.4f}, "
                        "Average pro metric score:{:.4f}, Time: {:d}ms/batch".
                            format(epoch+1, int(batch_id), total_sp_loss/(batch_id+1), total_unpro_metric_loss/(batch_id+1),
                                   total_pro_metric_loss/(batch_id+1), int(1000*(time.time()-start_time)/(batch_id+1))))
            return total_sp_loss / (batch_id+1), total_pro_metric_loss / (batch_id)

    def update_params(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_norm >= 0.0:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.gradient_norm)
        has_nan_inf = 0
        for params in self.net.parameters():
            if params.requires_grad:
                has_nan_inf += torch.sum(torch.isnan(params.grad))
                has_nan_inf += torch.sum(torch.isinf(params.grad))
        if has_nan_inf == 0:
            self.optimizer.step()

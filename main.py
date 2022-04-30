import os
import torch
import importlib
import logging
import numpy as np
import random
import time
import argparse
from dataset.data import InstanceDataset, InstanceDataloader
from solver import Solver
import warnings
import toml
from utils.utils import json_extraction, numParams, logger_print
warnings.filterwarnings('ignore')

# fix random seed
def setup_seed(seed):
    """
    set up random seed
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(config):
    # define seeds
    setup_seed(config["reproducibility"]["seed"])

    # set logger
    if not os.path.exists(config["path"]["logging_path"]):
        os.makedirs(config["path"]["logging_path"])
    logging.basicConfig(filename=config["path"]["logging_path"] + "/" + config["save"]["logger_filename"],
                        filemode='w',
                        level=logging.INFO,
                        format="%(message)s")
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger_print(f"start logging time:\t{start_time}")

    # set gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu_id) for gpu_id in config["gpu"]["gpu_ids"]])
    logger_print(f"gpus: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # set network
    net_choice = config["net"]["choice"]
    module = importlib.import_module(config["net"]["path"])
    net_args = config["net"][net_choice]["args"]
    net = getattr(module, config["net"]["classname"])(**net_args)
    logger_print(f"The number of trainable parameters: {numParams(net)}")

    # paths generation
    if not os.path.exists(config["path"]["checkpoint_load_path"]):
        os.makedirs(config["path"]["checkpoint_load_path"])
    if not os.path.exists(config["path"]["loss_save_path"]):
        os.makedirs(config["path"]["loss_save_path"])
    if not os.path.exists(config["path"]["model_best_path"]):
        os.makedirs(config["path"]["model_best_path"])

    # save filename
    save_name_dict = {}
    save_name_dict["loss_filename"] = config["save"]["loss_filename"]
    save_name_dict["best_model_filename"] = config["save"]["best_model_filename"]
    save_name_dict["checkpoint_filename"] = config["save"]["checkpoint_filename"]

    # determine file json
    train_mix_json = json_extraction(config["path"]["train"]["mix_file_path"], config["dataset"]["train"]["json_path"], "mix")
    val_mix_json = json_extraction(config["path"]["val"]["mix_file_path"], config["dataset"]["val"]["json_path"], "mix")

    # define train/validation
    train_dataset = InstanceDataset(mix_file_path=config["path"]["train"]["mix_file_path"],
                                    target_file_path=config["path"]["train"]["target_file_path"],
                                    mix_json_path=train_mix_json,
                                    is_variance_norm=config["signal"]["is_variance_norm"],
                                    is_chunk=config["signal"]["is_chunk"],
                                    chunk_length=config["signal"]["chunk_length"],
                                    sr=config["signal"]["sr"],
                                    batch_size=config["dataset"]["train"]["batch_size"],
                                    is_shuffle=config["dataset"]["train"]["is_shuffle"])
    val_dataset = InstanceDataset(mix_file_path=config["path"]["val"]["mix_file_path"],
                                  target_file_path=config["path"]["val"]["target_file_path"],
                                  mix_json_path=val_mix_json,
                                  is_variance_norm=config["signal"]["is_variance_norm"],
                                  is_chunk=config["signal"]["is_chunk"],
                                  chunk_length=config["signal"]["chunk_length"],
                                  sr=config["signal"]["sr"],
                                  batch_size=config["dataset"]["val"]["batch_size"],
                                  is_shuffle=config["dataset"]["val"]["is_shuffle"])
    train_dataloader = InstanceDataloader(train_dataset,
                                          **config["dataloader"]["train"])
    val_dataloader = InstanceDataloader(val_dataset,
                                        **config["dataloader"]["val"])

    # define optimizer
    if config["optimizer"]["name"] == "adam":
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=config["optimizer"]["lr"],
            betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
            weight_decay=config["optimizer"]["l2"])

    data = {'train_loader': train_dataloader, 'val_loader': val_dataloader}
    solver = Solver(data, net, optimizer, save_name_dict, config)
    solver.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", "--config", type=str, required=False, default="configs/train_config.toml",
                        help="toml format")
    args = parser.parse_args()
    config = toml.load(args.C)
    print(config)
    main(config)

import time
from collections import deque
from itertools import chain
import numpy as np
import torch
import sys
import os
import copy
from scipy import stats
import torch.nn as nn
from src.utils import get_argparser
from src.encoders_ICA import NatureCNN

import pandas as pd
import datetime
from src.lstm_attn import subjLSTM
from src.All_Architecture import combinedModel

# import torchvision.models.resnet_conv1D as models
# from tensorboardX import SummaryWriter

from src.graph_the_works_fMRI import the_works_trainer
import matplotlib.pyplot as plt
import nibabel as nib
import h5py
import math
from copy import copy
import matplotlib.colors as colors

import torch.nn.utils.rnn as tn


from src.ts_data import load_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
import wandb


def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


def train_encoder(args, k: int, trial: int):

    wdb1 = "wandb_new"
    wpath1 = os.path.join(os.getcwd(), wdb1)

    current_gain = 1
    args.gain = current_gain

    features, labels = load_dataset(args.ds)

    n_regions = features.shape[1]
    sample_y = 1
    subjects = features.shape[0]
    tc = features.shape[2]
    window_shift = 1

    samples_per_subject = int(tc / sample_y)

    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        print(torch.cuda.device_count())
        device = torch.device("cuda:0")
        device2 = torch.device("cuda:0")

    else:
        device = torch.device("cpu")
        device2 = torch.device("cpu")
    print("device = ", device)
    print("device = ", device2)

    for t in range(subjects):
        for r in range(n_regions):
            features[t, r, :] = stats.zscore(features[t, r, :])

    new_features = np.zeros((subjects, samples_per_subject, n_regions, sample_y))
    for i in range(subjects):
        for j in range(samples_per_subject):
            new_features[i, j, :, :] = features[
                i, :, (j * window_shift) : (j * window_shift) + sample_y
            ]

    features = new_features

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf.get_n_splits(features, labels)

    train_index, test_index = list(skf.split(features, labels))[k]

    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=42 + trial,
        stratify=y_train,
    )

    tr_eps = torch.from_numpy(X_train).float()
    val_eps = torch.from_numpy(X_val).float()
    test_eps = torch.from_numpy(X_test).float()

    tr_labels = torch.from_numpy(y_train).int()
    val_labels = torch.from_numpy(y_val).int()
    test_labels = torch.from_numpy(y_test).int()

    tr_labels = tr_labels.to(device)
    val_labels = val_labels.to(device)
    test_labels = test_labels.to(device)

    tr_eps = tr_eps.to(device)
    val_eps = val_eps.to(device)
    test_eps = test_eps.to(device)

    observation_shape = features.shape
    if args.model_type == "graph_the_works":
        print("obs shape", observation_shape[3])
        encoder = NatureCNN(observation_shape[3], args, observation_shape[2])
        encoder.to(device)
        lstm_model = subjLSTM(
            device,
            sample_y,
            args.lstm_size,
            num_layers=args.lstm_layers,
            freeze_embeddings=True,
            gain=current_gain,
            bidrection=True,
        )

        dir = ""
        if args.pre_training == "DECENNT":
            dir = "Pre_Trained/DECENNT/UsingHCP500TP/model.pt"
            args.oldpath = wpath1 + "/Pre_Trained/DECENNT/UsingHCP500TP"

    complete_model = combinedModel(
        encoder,
        lstm_model,
        samples_per_subject,
        time=tc,
        gain=current_gain,
        PT=args.pre_training,
        exp=args.exp,
        device_one=device,
        oldpath=args.oldpath,
        n_regions=n_regions,
        device_two=device2,
        device_zero=device2,
        device_extra=device2,
    )

    config = {}
    config.update(vars(args))

    config["obs_space"] = observation_shape  # weird hack
    if args.method == "graph_the_works":
        trainer = the_works_trainer(
            complete_model,
            config,
            device=device2,
            device_encoder=device,
            tr_labels=tr_labels,
            val_labels=val_labels,
            test_labels=test_labels,
            trial=str(trial),
            crossv=str(k),
            gtrial=str(trial),
        )

    else:
        assert False, "method {} has no trainer".format(args.method)

    wandb_logger: wandb.run = wandb.init(
        project=f"{args.prefix}-experiment-dice-{args.ds}",
        name=f"k_{k}-trial_{trial}",
        save_code=True,
    )

    (
        test_accuracy,
        test_score,
        test_loss,
        e,
        test_precision,
        test_recall,
        edge_weights,
    ) = trainer.train(tr_eps, val_eps, test_eps)
    results = {
        "test_accuracy": test_accuracy,
        "test_score": test_score,
        "test_loss": test_loss,
        # "e": e,
        # "test_precision": test_precision,
        # "test_recall": test_recall,
        # "edge_weights": edge_weights,
    }

    wandb_logger.log(results)
    wandb_logger.finish()

    result_csv = os.path.join(args.path, "test_results.csv")
    df = pd.DataFrame(results, index=[0])
    with open(result_csv, "a") as f:
        df.to_csv(f, header=f.tell() == 0, index=False)

    print("AUC: ", test_score)


if __name__ == "__main__":
    CUDA_LAUNCH_BLOCKING = "1"
    # torch.manual_seed(33)
    # np.random.seed(33)
    parser = get_argparser()
    args = parser.parse_args()
    tags = ["pretraining-only"]
    config = {}
    config.update(vars(args))
    for k in range(5):
        for trial in range(10):
            train_encoder(args, k, trial)

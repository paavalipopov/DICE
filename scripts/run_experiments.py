import numpy as np
import torch
import os
from scipy import stats
from src.utils import get_argparser
from src.encoders_ICA import NatureCNN

import pandas as pd
from src.lstm_attn import subjLSTM
from src.All_Architecture import combinedModel

from src.graph_the_works_fMRI import the_works_trainer


from src.ts_data import load_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
import wandb


def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


def train_encoder(args):

    wdb1 = "wandb_new"
    wpath1 = os.path.join(os.getcwd(), wdb1)

    current_gain = 1
    args.gain = current_gain

    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        print(torch.cuda.device_count())
        device = torch.device("cuda:0")
        device2 = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        device2 = torch.device("cpu")

    print("device = ", device)
    print("device2 = ", device2)

    # obtain data, set params for reshaping
    features, labels = load_dataset(args.ds)

    tcs = []
    tcs.append(features.shape[2])
    for dataset in args.test_ds:
        local_features, _ = load_dataset(dataset)
        tcs.append(local_features.shape[2])

    # find minimal time-course
    min_tc = np.min(tcs)
    features = features[:, :, :min_tc]

    # get rid of invalid data (NaNs, infs)
    features[features != features] = 0

    n_regions = features.shape[1]
    sample_y = 1
    subjects = features.shape[0]
    tc = features.shape[2]
    window_shift = 1

    samples_per_subject = int(tc / sample_y)

    # z-score data
    for t in range(subjects):
        for r in range(n_regions):
            features[t, r, :] = stats.zscore(features[t, r, :])

    # reshape data into windows
    new_features = np.zeros((subjects, samples_per_subject, n_regions, sample_y))
    for i in range(subjects):
        for j in range(samples_per_subject):
            new_features[i, j, :, :] = features[
                i, :, (j * window_shift) : (j * window_shift) + sample_y
            ]

    features = new_features

    #### extra test datasets
    extra_test_eps = []
    extra_test_labels = {}
    for dataset in args.test_ds:
        local_features, local_labels = load_dataset(dataset)
        local_features = local_features[:, :, :min_tc]
        local_features[local_features != local_features] = 0

        n_regions = local_features.shape[1]
        sample_y = 1
        subjects = local_features.shape[0]
        tc = local_features.shape[2]
        window_shift = 1

        samples_per_subject = int(tc / sample_y)

        # z-score data
        for t in range(subjects):
            for r in range(n_regions):
                local_features[t, r, :] = stats.zscore(local_features[t, r, :])

        # reshape data into windows
        new_features = np.zeros((subjects, samples_per_subject, n_regions, sample_y))
        for i in range(subjects):
            for j in range(samples_per_subject):
                new_features[i, j, :, :] = local_features[
                    i, :, (j * window_shift) : (j * window_shift) + sample_y
                ]

        local_features = torch.from_numpy(new_features).float()

        extra_test_eps.append(
            {
                "name": dataset,
                "eps": local_features.to(device),
            }
        )

        extra_test_labels[dataset] = torch.from_numpy(local_labels).int().to(device)
    #######

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in range(5):
        for trial in range(10):
            # split into train/val/test
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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

            # Usman's code

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

            result_dir = os.path.join(
                args.path,
                f"{args.prefix}-experiment-dice-{args.ds}/{k:02d}/{trial:04d}",
            )
            os.makedirs(result_dir, exist_ok=True)
            config["path"] = result_dir

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
                    extra_test_labels=extra_test_labels,
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
                extra_tests,
            ) = trainer.train(tr_eps, val_eps, test_eps, extra_test_eps)
            results = {
                "test_accuracy": test_accuracy,
                "test_score": test_score,
                "test_loss": test_loss,
            }
            for dataset in extra_tests:
                results[f"{dataset}_test_accuracy"] = extra_tests[dataset][
                    "test_accuracy"
                ]
                results[f"{dataset}_test_score"] = extra_tests[dataset]["test_score"]
                results[f"{dataset}_test_loss"] = extra_tests[dataset]["test_loss"]

            wandb_logger.log(results)
            wandb_logger.finish()

            result_csv = os.path.join(
                result_dir,
                "test_results.csv",
            )

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

    train_encoder(args)

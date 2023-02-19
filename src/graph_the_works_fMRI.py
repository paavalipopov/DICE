import random

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler, SequentialSampler
from .utils import (
    calculate_accuracy,
    Cutout,
    calculate_accuracy_by_labels,
    calculate_FP,
    calculate_FP_Max,
)
from .newtrainer import Trainer
from src.utils import EarlyStopping, EarlyStoppingACC, EarlyStoppingACC_and_Loss

# from torchvision import transforms
import matplotlib.pylab as plt
import matplotlib.pyplot as pl

# import torchvision.transforms.functional as TF
import torch.nn.utils.rnn as tn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import torch.nn.utils.rnn as tn
from sklearn.linear_model import LogisticRegression
from itertools import combinations, product
from torch.nn import TripletMarginLoss
from torch import int as tint, long, short, Tensor
from random import sample
from sys import maxsize
from collections import Counter
import csv
import time
import math


class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class the_works_trainer(Trainer):
    def __init__(
        self,
        model,
        config,
        device,
        device_encoder,
        tr_labels,
        val_labels,
        test_labels,
        extra_test_labels,
        test_labels2="",
        trial="",
        crossv="",
        gtrial="",
        tr_FNC="tr_FNC",
        val_FNC="val_FNC",
        test_FNC="test_FNC",
    ):
        super().__init__(model, device)
        self.config = config
        self.device_encoder = device_encoder
        self.tr_labels = tr_labels
        self.test_labels = test_labels
        self.extra_test_labels = extra_test_labels
        self.test_labels2 = test_labels2
        self.val_labels = val_labels
        self.tr_FNC = tr_FNC
        self.val_FNC = val_FNC
        self.test_FNC = test_FNC
        self.criterion2 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.78))
        self.patience = self.config["patience"]
        self.dropout = nn.Dropout(0.65).to(device)
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.sample_number = config["sample_number"]
        self.path = config["path"]
        self.oldpath = config["oldpath"]
        self.fig_path = config["fig_path"]
        self.p_path = config["p_path"]
        self.PT = config["pre_training"]
        self.device = device
        self.gain = config["gain"]
        (
            self.train_epoch_loss,
            self.train_batch_loss,
            self.eval_epoch_loss,
            self.eval_batch_loss,
            self.eval_batch_accuracy,
            self.train_epoch_accuracy,
        ) = ([], [], [], [], [], [])
        self.train_epoch_roc, self.eval_epoch_roc = [], []
        self.eval_epoch_CE_loss, self.eval_epoch_E_loss, self.eval_epoch_lstm_loss = (
            [],
            [],
            [],
        )
        self.test_accuracy = 0.0
        self.test_auc = 0.0
        self.test_precision = 0.0
        self.test_recall = 0.0
        self.test_loss = 0.0
        self.n_heads = 1
        self.edge_weights = ""
        self.temporal_edge_weights = ""
        self.edge_weights_sum = ""
        self.attention_region = ""
        self.attention_time = ""
        self.attention_weights = ""
        self.attention_component = ""
        self.attention_time_embeddings = ""
        self.FNC = ""
        self.trials = trial
        self.gtrial = gtrial
        self.exp = config["exp"]
        self.cv = crossv
        self.test_targets = ""
        self.test_targets2 = ""
        self.test_predictions = ""
        self.regions_selected = ""
        self.means_labels = ""
        self.loss_criteria = nn.L1Loss()
        self.triplet_loss_function = TripletMarginLoss(margin=0.5)
        self.lr = config["lr"]
        self.dropout = nn.Dropout(0.65).to(self.device)

        if self.exp in ["UFPT", "NPT"]:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

        self.early_stopper = EarlyStopping(
            "self.model_backup",
            patience=self.patience,
            verbose=False,
            wandb="self.wandb",
            name="model",
            path=self.path,
            trial=self.trials,
        )

    def find_value_ids(self, it, value):
        """
        Args:
            it: list of any
            value: query element

        Returns:
            indices of the all elements equal x0
        """
        if isinstance(it, np.ndarray):
            inds = list(np.where(it == value)[0])
        else:  # could be very slow
            inds = [i for i, el in enumerate(it) if el == value]
        return inds

    def _check_input_labels(self, labels):
        """
        The input must satisfy the conditions described in
        the class documentation.

        Args:
            labels: labels of the samples in the batch
        """
        labels_counter = Counter(labels)
        assert all(n > 1 for n in labels_counter.values())
        assert len(labels_counter) > 1

    def mysample(self, features, labels):
        if isinstance(labels, Tensor):
            labels = labels.tolist()
        self._check_input_labels(labels)
        ids_anchor, ids_pos, ids_neg = self.mysample2(features, labels=labels)
        return features[ids_anchor], features[ids_pos], features[ids_neg]

    def mysample2(self, features, labels):
        num_labels = len(labels)

        triplets = []
        for label in set(labels):
            ids_pos_cur = set(self.find_value_ids(labels, label))
            ids_neg_cur = set(range(num_labels)) - ids_pos_cur

            pos_pairs = list(combinations(ids_pos_cur, r=2))

            tri = [(a, p, n) for (a, p), n in product(pos_pairs, ids_neg_cur)]
            triplets.extend(tri)

        triplets = sample(triplets, min(len(triplets), maxsize))
        ids_anchor, ids_pos, ids_neg = zip(*triplets)

        return list(ids_anchor), list(ids_pos), list(ids_neg)

    def TripletLoss(self, features, labels):
        """
        Args:
            features: features with shape [batch_size, features_dim]
            labels: labels of samples having batch_size elements

        Returns: loss value

        """

        features_anchor, features_positive, features_negative = self.mysample(
            features=features, labels=labels
        )

        loss = self.triplet_loss_function(
            anchor=features_anchor,
            positive=features_positive,
            negative=features_negative,
        )
        return loss

    def generate_batch(self, episodes, mode):
        if self.sample_number == 0:
            total_steps = sum([len(e) for e in episodes])
        else:
            total_steps = self.sample_number

        if mode == "train" or mode == "eval":
            BS = self.batch_size
        else:
            BS = self.batch_size  # math.ceil(episodes.shape[0]/5)

        if mode == "seq":
            sampler = BatchSampler(
                SequentialSampler(range(len(episodes))), BS, drop_last=False
            )
        else:
            sampler = BatchSampler(
                RandomSampler(range(len(episodes)), replacement=False),
                BS,
                drop_last=False,
            )

        for indices in sampler:

            episodes_batch = [episodes[x, :, :, :] for x in indices]

            ts_number = torch.LongTensor(indices)
            i = 0
            sx = []

            yield torch.stack(episodes_batch).to(self.device_encoder), ts_number.to(
                self.device_encoder
            )

    def get_prediction_loss(self, preds, target, variance=5e-5, add_const=False):
        neg_log_p = (preds - target) ** 2 / (2 * variance)
        if add_const:
            const = 0.5 * np.log(2 * np.pi * variance)
            neg_log_p += const
        return neg_log_p.sum() / (target.size(0) * target.size(1))

    def mixup_data(self, x, y, alpha=1.0, device="cuda"):

        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        mixed_y = lam * y + (1 - lam) * y[index]

        ones = mixed_y >= 0.5
        zeros = mixed_y < 0.5
        mixed_y[zeros] = 0
        mixed_y[ones] = 1
        return mixed_x, y_a, y_b, mixed_y, lam

    def do_one_epoch(self, epoch, episodes, mode, test_name=None):

        (
            epoch_loss,
            epoch_loss2,
            epoch_loss3,
            accuracy,
            steps,
            epoch_acc,
            epoch_roc,
            epoch_roc2,
            epoch_prec,
            epoch_recall,
        ) = (0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        epoch_CE_loss, epoch_E_loss, epoch_lstm_loss = (
            0.0,
            0.0,
            0.0,
        )
        accuracy1, accuracy2, accuracy, FP = 0.0, 0.0, 0.0, 0.0
        (
            epoch_loss_mi,
            epoch_loss_mse,
            epoch_accuracy,
            epoch_accuracy2,
            epoch_FP,
            epoch_total_loss,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        all_logits = ""

        data_generator = self.generate_batch(episodes, mode)
        for sx, ts_number in data_generator:
            FNC = ""

            loss = 0.0
            loss2 = 0.0
            loss3 = 0.0
            diag = 0.0
            ndiag = 0.0
            lam = 0.0
            loss_total, loss_pred = 0.0, 0.0
            CE_loss, E_loss, lstm_loss = 0.0, 0.0, 0.0
            targets = ""
            targets2 = ""
            if mode == "train":
                targets = self.tr_labels[ts_number]
                # FNC = self.tr_FNC[ts_number,:]

            elif mode == "eval":
                targets = self.val_labels[ts_number]
                # FNC = self.val_FNC[ts_number, :]

            else:
                if test_name is None:
                    targets = self.test_labels[ts_number]
                else:
                    targets = self.extra_test_labels[test_name][ts_number]

            targets = targets.long()
            targets = targets.to(self.device)
            sx = sx.to(self.device)

            logits, kl_loss, FC, FC_temporal = self.model(
                sx, targets, mode, self.device, epoch
            )

            loss = F.cross_entropy(logits, targets)

            # loss =  loss + loss_mse + loss2
            if mode == "train" or mode == "eval":
                # loss = loss + kl_loss
                loss, CE_loss, E_loss, lstm_loss = self.add_regularization(loss)

            t = time.time()

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()

            if all_logits == "":
                all_logits = logits.detach()
                all_targets = targets.detach()
            else:
                all_logits = torch.cat((all_logits, logits.detach()), dim=0)
                all_targets = torch.cat((all_targets, targets.detach()), dim=0)

            if mode == "train" or mode == "eval":
                epoch_E_loss += E_loss

            if mode == "test":
                if self.edge_weights == "":

                    self.edge_weights = FC.detach()

                else:
                    self.edge_weights = torch.cat(
                        (self.edge_weights, FC.detach()), dim=0
                    )

            del loss
            del loss2
            # del loss3
            del targets
            del diag
            del FC
            del logits
            del loss_total

            steps += 1

        accuracy, roc, pred, prec, recall = self.acc_and_auc(
            all_logits, mode, all_targets
        )
        if mode != "train":
            epoch_roc += roc * steps
            epoch_prec += prec * steps
            epoch_recall += recall * steps

        epoch_accuracy += accuracy.detach().item() * steps

        # t = time.time()
        if mode == "eval":
            self.eval_batch_accuracy.append(epoch_accuracy / steps)
            self.eval_epoch_loss.append(epoch_loss / steps)
            self.eval_epoch_roc.append(epoch_roc / steps)
            self.eval_epoch_CE_loss.append(epoch_CE_loss / steps)
            self.eval_epoch_E_loss.append(epoch_E_loss / steps)
            self.eval_epoch_lstm_loss.append(epoch_lstm_loss / steps)
        elif mode == "train":
            self.train_epoch_loss.append(epoch_loss / steps)
            self.train_epoch_accuracy.append(epoch_accuracy / steps)
        if epoch % 1 == 0:
            self.log_results(
                epoch,
                epoch_loss / steps,
                epoch_loss / steps,
                epoch_loss / steps,
                epoch_accuracy / steps,
                epoch_accuracy2 / steps,
                epoch_roc / steps,
                epoch_roc2 / steps,
                epoch_prec / steps,
                epoch_recall / steps,
                prefix=mode,
            )
        if mode == "eval" and epoch > -1:
            best_on = epoch_accuracy / steps
            self.early_stopper(epoch_loss / steps, best_on, self.model, 0, epoch=epoch)
        if mode == "test":
            if test_name is None:
                self.test_accuracy = epoch_accuracy / steps
                self.test_auc = epoch_roc / steps
                self.test_loss = epoch_loss / steps
                self.test_precision = epoch_prec / steps
                self.test_recall = epoch_recall / steps
                self.test_targets = all_targets
            else:
                self.extra_test_log[test_name] = {}
                self.extra_test_log[test_name]["test_accuracy"] = epoch_accuracy / steps
                self.extra_test_log[test_name]["test_score"] = epoch_roc / steps
                self.extra_test_log[test_name]["test_loss"] = epoch_loss / steps

        return epoch_loss / steps

    def acc_and_auc(self, logits, mode, targets):

        sig = torch.softmax(logits, dim=1)
        values, indices = sig.max(1)

        roc = 0.0
        prec = 0.0
        rec = 0.0
        acc = 0.0

        if 1 in targets and 0 in targets:
            if mode != "train":
                y_scores = (sig.detach()[:, 1]).float()
                roc = roc_auc_score(targets.to("cpu"), y_scores.to("cpu"))
                prec = precision_score(targets.to("cpu"), indices.to("cpu"))
                rec = recall_score(targets.to("cpu"), indices.to("cpu"))
        accuracy = calculate_accuracy_by_labels(indices, targets)

        return accuracy, roc, indices, prec, rec

    def add_regularization(self, loss, ortho_loss=0.0):
        reg = 1e-6
        E_loss = 0.0
        lstm_loss = torch.zeros(1).to(self.device)
        orth_loss = torch.zeros(1).to(self.device)
        attn_loss = 0.0
        mha_loss = 0.0
        CE_loss = loss
        encoder_loss = torch.zeros(1).to(self.device)

        for name, param in self.model.gta_embed.named_parameters():
            if "bias" not in name:
                lstm_loss += reg * torch.norm(param, p=1)

        for name, param in self.model.gta_attend.named_parameters():
            if "bias" not in name:
                lstm_loss += reg * torch.norm(param, p=1)

        loss = loss + lstm_loss.to(self.device)  # + encoder_loss
        return loss, CE_loss, E_loss, lstm_loss

    def validate(self, val_eps):

        model_dict = torch.load(
            os.path.join(self.p_path, "encoder" + self.trials + ".pt"),
            map_location=self.device,
        )
        self.encoder.load_state_dict(model_dict)
        self.encoder.eval()
        self.encoder.to(self.device)

        model_dict = torch.load(
            os.path.join(self.p_path, "lstm" + self.trials + ".pt"),
            map_location=self.device,
        )
        self.lstm.load_state_dict(model_dict)
        self.lstm.eval()
        self.lstm.to(self.device)

        mode = "eval"
        self.do_one_epoch(0, val_eps, mode)
        return self.test_auc

    def load_model_and_test(self, tst_eps, test_name=None):
        print("Best model was", self.early_stopper.epoch_saved)
        model_dict = torch.load(
            os.path.join(self.path, "model" + self.trials + ".pt"),
            map_location=self.device,
        )
        self.model.load_state_dict(model_dict)
        self.model.eval()

        mode = "test"
        self.do_one_epoch(0, tst_eps, mode, test_name)

    def save_loss_and_auc(self):

        with open(
            os.path.join(self.path, "all_data_information" + self.trials + ".csv"),
            "w",
            newline="",
        ) as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            self.train_epoch_loss.insert(0, "train_epoch_loss")
            wr.writerow(self.train_epoch_loss)

            self.train_epoch_accuracy.insert(0, "train_epoch_accuracy")
            wr.writerow(self.train_epoch_accuracy)

            self.eval_epoch_loss.insert(0, "eval_epoch_loss")
            wr.writerow(self.eval_epoch_loss)

            self.eval_batch_accuracy.insert(0, "eval_batch_accuracy")
            wr.writerow(self.eval_batch_accuracy)

            self.eval_epoch_roc.insert(0, "eval_epoch_roc")
            wr.writerow(self.eval_epoch_roc)

            self.eval_epoch_CE_loss.insert(0, "eval_epoch_CE_loss")
            wr.writerow(self.eval_epoch_CE_loss)

            self.eval_epoch_E_loss.insert(0, "eval_epoch_E_loss")
            wr.writerow(self.eval_epoch_E_loss)

            self.eval_epoch_lstm_loss.insert(0, "eval_epoch_lstm_loss")
            wr.writerow(self.eval_epoch_lstm_loss)

    def train(self, tr_eps, val_eps, tst_eps, extra_test_eps):
        print("lr = ", self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=4, factor=0.5, cooldown=0, verbose=True
        )

        if self.PT in ["DECENNT"]:
            if self.exp in ["UFPT", "FPT"]:
                print("in ufpt and fpt")
                model_dict = torch.load(
                    os.path.join(self.oldpath, "model" + ".pt"),
                    map_location=self.device,
                )
                self.model.load_state_dict(model_dict)
                self.model.to(self.device)
            else:
                self.model.init_weight(PT=self.exp)
        #
        else:
            self.model.init_weight(PT=self.exp)

        saved = 0

        for e in range(self.epochs):
            # print(self.exp)
            if self.exp in ["UFPT", "NPT"]:
                self.model.train()

            else:
                self.model.eval()

            mode = "train"

            val_loss = self.do_one_epoch(e, tr_eps, mode)
            self.model.eval()
            mode = "eval"
            # print("====================================VALIDATION START===============================================")
            val_loss = self.do_one_epoch(e, val_eps, mode)

            scheduler.step(val_loss)

            if self.early_stopper.early_stop:
                self.early_stopper(0, 0, self.model, 1, epoch=e)
                saved = 1
                break

        if saved == 0:
            self.early_stopper(0, 0, self.model, 1, epoch=e)
            saved = 1

        self.save_loss_and_auc()
        self.load_model_and_test(tst_eps)

        self.extra_test_log = {}
        for dataset in extra_test_eps:
            self.load_model_and_test(dataset["eps"], dataset["name"])

        return (
            self.test_accuracy,
            self.test_auc,
            self.test_loss,
            self.extra_test_log,
        )

    def log_results(
        self,
        epoch_idx,
        epoch_loss2,
        epoch_loss3,
        epoch_loss,
        epoch_test_accuracy,
        epoch_accuracy2,
        epoch_roc,
        epoch_roc2,
        prec,
        recall,
        prefix="",
    ):
        print(
            "{} Test fold: {}, Trial: {}, Epoch: {}, Loss: {}, Accuracy: {}, roc: {}, prec: {}, recall:{}".format(
                prefix.capitalize(),
                self.cv,
                self.trials,
                epoch_idx,
                epoch_loss,
                epoch_test_accuracy,
                epoch_roc,
                prec,
                recall,
            )
        )

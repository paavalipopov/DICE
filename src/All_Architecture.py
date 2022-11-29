import torch
import torch.nn as nn
import torch.nn.utils.rnn as tn
from torch.autograd import Variable
import os
import torch.nn.functional as F
import numpy as np
import scipy
import time
import copy
import random
from .utils import init
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


class combinedModel(nn.Module):
    """Bidirectional LSTM for classifying subjects."""

    def __init__(
        self,
        encoder,
        lstm,
        samples_per_subject,
        time,
        gain=0.1,
        PT="",
        exp="UFPT",
        device_one="cuda",
        oldpath="",
        k=10,
        n_regions=100,
        device_two="",
        device_zero="",
        device_extra="",
    ):

        super().__init__()
        self.encoder = encoder

        self.lstm = lstm
        self.gain = gain
        # self.graph = graph
        self.samples_per_subject = 1
        self.n_clusters = 4
        self.w = 1
        self.n_regions = n_regions
        self.n_regions_after = n_regions
        self.PT = PT
        self.exp = exp
        self.device_one = device_one
        self.device_two = device_two
        self.oldpath = oldpath
        # self.time_points=155
        self.time_points = time
        self.n_heads = 1
        self.attention_embedding = 48 * self.n_heads
        self.k = 10000  # k
        self.upscale = 0.05  # 1##2#1#0.25 #HCP.005 and FBIRN region
        self.upscale2 = 0.5  # 1#0.5
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        self.temperature = 2

        self.relu = torch.nn.ReLU()
        self.HS = torch.nn.Hardsigmoid()
        self.HW = torch.nn.Hardswish()
        self.selu = torch.nn.SELU()
        self.celu = torch.nn.CELU()
        self.tanh = torch.nn.Tanh()
        self.softplus = torch.nn.Softplus(threshold=70)

        self.gta_embed = nn.Sequential(
            nn.Linear(
                self.n_regions * self.n_regions,
                round(self.upscale * self.n_regions * self.n_regions),
            ),
        ).to(self.device_two)

        self.gta_norm = nn.Sequential(
            nn.BatchNorm1d(round(self.upscale * self.n_regions * self.n_regions)),
            nn.ReLU(),
        ).to(self.device_two)

        self.gta_attend = nn.Sequential(
            nn.Linear(
                round(self.upscale * self.n_regions * self.n_regions),
                round(self.upscale2 * self.n_regions * self.n_regions),
            ),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(round(self.upscale2 * self.n_regions * self.n_regions), 1),
        ).to(self.device_two)

        self.gta_dropout = nn.Dropout(0.35)

        self.key_layer = nn.Sequential(
            nn.Linear(
                self.samples_per_subject * self.lstm.output_dim,
                self.samples_per_subject * self.attention_embedding,
            ),
        ).to(self.device_one)

        self.value_layer = nn.Sequential(
            nn.Linear(
                self.samples_per_subject * self.lstm.output_dim,
                self.samples_per_subject * self.attention_embedding,
            ),
        ).to(self.device_one)

        self.query_layer = nn.Sequential(
            nn.Linear(
                self.samples_per_subject * self.lstm.output_dim,
                self.samples_per_subject * self.attention_embedding,
            ),
        ).to(self.device_one)

        # self.means_to_higher_projection = nn.Sequential(
        #     nn.Linear(64,256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.n_regions_after * self.n_regions_after),
        #
        # )  # .to(self.device_one)

        self.multihead_attn = nn.MultiheadAttention(
            self.samples_per_subject * self.attention_embedding, self.n_heads
        ).to(self.device_one)

    def init_weight(self, PT="UFPT"):
        print(self.gain)
        print("init" + PT)
        # return
        if PT == "NPT":
            for name, param in self.query_layer.named_parameters():
                if "weight" in name:
                    nn.init.kaiming_normal_(param, mode="fan_in")
                # param = param + torch.abs(torch.min(param))

            for name, param in self.key_layer.named_parameters():
                if "weight" in name:
                    nn.init.kaiming_normal_(param, mode="fan_in")
                # param = param + torch.abs(torch.min(param))
            for name, param in self.value_layer.named_parameters():
                if "weight" in name:
                    nn.init.kaiming_normal_(param, mode="fan_in")
                # param = param + torch.abs(torch.min(param))
            for name, param in self.multihead_attn.named_parameters():
                if "weight" in name:
                    nn.init.kaiming_normal_(param, mode="fan_in")
                # param = param + torch.abs(torch.min(param))

                # param = param + torch.abs(torch.min(param))

            for name, param in self.gta_embed.named_parameters():
                # print('name = ',name)
                if "weight" in name and param.dim() > 1:
                    nn.init.kaiming_normal_(param, mode="fan_in")
                # with torch.no_grad():
                #     param.add_(torch.abs(torch.min(param)))
                # print(torch.min(param))

            for name, param in self.gta_attend.named_parameters():
                # print(name)

                if "weight" in name and param.dim() > 1:
                    # print(name)
                    # print(param.min())
                    nn.init.kaiming_normal_(param, mode="fan_in")

        for name, param in self.encoder.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")

    def loadModels(self):
        if self.PT in ["milc-fMRI", "variable-attention", "two-loss-milc"]:
            if self.exp in ["UFPT", "FPT"]:
                print("in ufpt and fpt")
                model_dict = torch.load(
                    os.path.join(self.oldpath, "model" + ".pt"),
                    map_location=self.device_one,
                )
                self.lstm.load_state_dict(model_dict)
                # self.model.lstm.to(self.device_one)

    def spatial_attention(self, inputs):

        inputs = inputs.squeeze()
        # print(inputs.shape)
        weights = self.attn_spatial(
            inputs.reshape(inputs.shape[0] * inputs.shape[1], -1)
        )
        weights = weights.squeeze().reshape(-1, self.n_regions)
        weights = weights.unsqueeze(2).repeat(1, 1, self.time_points)

        inputs = weights * inputs

        return inputs.unsqueeze(3), weights

    def gta_attention_embeddings(self, x, node_axis=1, dimension="time", mode="train"):
        if dimension == "time":

            x_readout = x.mean(node_axis, keepdims=True)
            x_readout = x - x_readout
            a = x_readout.shape[0]
            b = x_readout.shape[1]
            x_readout = x_readout.reshape(-1, x_readout.shape[2])
            x_embed = self.gta_norm_embeddings(self.gta_embed_embeddings(x_readout))
            x_graphattention = self.gta_attend_embeddings(x_embed).squeeze()
            x_graphattention = x_graphattention.reshape(a, b)

            return (x * (x_graphattention.unsqueeze(-1))).mean(node_axis)

    def gta_attention(self, x, node_axis=1, outputs="", dimension="time", mode="train"):
        if dimension == "time":

            x_readout = x.mean(node_axis, keepdim=True)
            x_readout = x * x_readout

            a = x_readout.shape[0]
            b = x_readout.shape[1]
            x_readout = x_readout.reshape(-1, x_readout.shape[2])
            x_embed = self.gta_norm(self.gta_embed(x_readout))
            # print('x embed min = ', torch.min(x_embed))
            x_graphattention = (self.gta_attend(x_embed).squeeze()).reshape(a, b)
            # print('x_graphattention min = ', torch.min(x_graphattention))
            # return
            # print('min = ', x_graphattention.min().item())
            # print('max = ', x_graphattention.max().item())
            x_graphattention = self.HW(x_graphattention.reshape(a, b))
            return (
                (x * (x_graphattention.unsqueeze(-1))).mean(node_axis),
                "FC2",
                x_graphattention,
            )

    def get_attention(self, outputs, type="weight"):

        if type == "weight":
            weights = self.attn_weight(
                outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)
            )
            weights = weights.squeeze().reshape(-1, self.time_points)

        elif type == "region":
            weights = self.attn_region(
                outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)
            )
            weights = weights.squeeze().reshape(-1, self.n_regions)
        elif type == "time":
            weights = self.attn_time(
                outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)
            )
            weights = weights.squeeze().reshape(-1, self.time_points)

        normalized_weights = weights  # torch.softmax(weights,dim=1)

        if type == "weight":

            v, indexes = torch.kthvalue(normalized_weights, 46, dim=1)
            v = v.unsqueeze(1).repeat(1, self.time_points)
            zeros = torch.zeros(v.size()).to(self.device_one)
            normalized_weights = torch.where(
                normalized_weights < v, normalized_weights, zeros
            )

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        # attn_applied = normalized_weights * outputs

        attn_applied = attn_applied.squeeze()

        return attn_applied, normalized_weights.unsqueeze(1)
        # else:
        #     return attn_applied

    def multi_head_attention(self, outputs, k, FNC="", FNC2=""):

        key = self.key_layer(outputs)
        value = self.value_layer(outputs)
        query = self.query_layer(outputs)

        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        query = query.permute(1, 0, 2)

        attn_output, attn_output_weights = self.multihead_attn(key, value, query)
        attn_output = attn_output.permute(1, 0, 2)

        attn_output_weights = attn_output_weights  # + FNC + FNC2
        return attn_output, attn_output_weights

    def multi_head_attention2(self, outputs, k, FNC="", FNC2=""):

        key = self.key_layer2(outputs)
        value = self.value_layer2(outputs)
        query = self.query_layer2(outputs)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        query = query.permute(1, 0, 2)

        attn_output, attn_output_weights = self.multihead_attn2(query, key, value)
        attn_output = attn_output.permute(1, 0, 2)

        attn_output_weights = attn_output_weights  # + FNC + FNC2
        return attn_output, attn_output_weights

    def get_topk_weights(self, weights):

        weights = weights.reshape(weights.shape[0], -1)
        sorted_weights = torch.argsort(weights, descending=True)
        top_k_weights = sorted_weights[:, : self.k]
        weights = weights.gather(1, sorted_weights)
        weights = weights[:, : self.k]
        return weights.reshape(weights.shape[0], -1), top_k_weights

    def get_lstm_loss(self, data, B):
        encoder_logits = self.lstm_decoder(data.permute(0, 1, 2, 3).reshape(B, -1))
        return encoder_logits

    def final_attention(self, data):
        weights = self.mlp2(data)
        data = weights * data

        return data, weights

    def forward(self, input, targets, mode="train", device="cpu", epoch=0, FNC=""):
        indices = ""
        # r = random.randint(0, 800)
        # sx = sx[:,r:r+400,:,:]

        B = input.shape[0]
        W = input.shape[1]
        R = input.shape[2]
        T = input.shape[3]
        input = input.reshape(B, 1, self.time_points, R, T)
        input = input.permute(1, 0, 2, 3, 4)
        (FC_logits), FC, FC_sum, FC_time_weights = 0.0, 0.0, 0.0, 0.0

        for sb in range(1):

            sx = input[sb, :, :, :, :]
            B = sx.shape[0]
            W = sx.shape[1]
            R = sx.shape[2]
            T = sx.shape[3]

            inputs = sx.permute(0, 2, 1, 3).contiguous()

            inputs = inputs.reshape(B * R, W, T)

            inputs = self.lstm(inputs)
            # inputs = inputs.to(self.device_one)

            inputs = inputs.reshape(B, R, W, inputs.shape[2])

            # lstm_output_before_MHA = self.lstm_decoder(torch.sum(inputs,dim=2).squeeze().reshape(B, -1))
            # print(inputs.shape)
            inputs = inputs.permute(2, 0, 1, 3).contiguous()
            # print(inputs.shape)
            inputs = inputs.reshape(W * B, R, self.lstm.output_dim)

            inputs = inputs.to(self.device_one)
            outputs, attn_weights = self.multi_head_attention(inputs, self.k)
            # print(self.device_two)

            attn_weights = attn_weights.to(self.device_two)
            # outputs = outputs.to(self.device_two)

            attn_weights = attn_weights.reshape(W, B, R, R)

            attn_weights = attn_weights.permute(1, 0, 2, 3).contiguous()

            attn_weights = attn_weights.reshape(B, W, -1)

            FC, FC2, FC_time_weights = self.gta_attention(
                attn_weights, dimension="time", mode=mode
            )

            FC = FC.reshape(B, R, R)
            FC_sum = torch.mean(attn_weights, dim=1).squeeze().reshape(B, R, R)

            if sb == 0:
                FC_logits = self.encoder((FC.unsqueeze(1)))
            else:
                FC_logits += self.encoder((FC.unsqueeze(1)))
            # outputs_logits = self.mlp2(outputs_emb.reshape(B, -1))

        if mode == "test":
            return (
                (FC_logits),
                0,
                FC,
                "fc_time",
            )

        return (
            (FC_logits),
            0,
            FC,
            "fc_time",
        )

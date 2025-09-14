import numpy as np
from sklearn.cluster import KMeans
from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from Dataloader import wavelet_transform

warnings.filterwarnings("ignore")

torch.set_printoptions(sci_mode=False, precision=2)


class GRUmodel(nn.Module):
    def __init__(self, inputsize=10, hiddensize=128, outsize=32):
        super(GRUmodel, self).__init__()
        self.rnn1 = nn.GRU(inputsize, hiddensize, 1, batch_first=True, )
        self.relu1 = nn.ReLU()
        self.rnn2 = nn.GRU(hiddensize, outsize, 1, batch_first=True, )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out, h = self.rnn1(x)
        out = self.relu1(out)
        out, h = self.rnn2(out)
        out = self.relu2(out)
        return out


class ExpertEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, expert1, expert2, x):
        out1 = expert1(x)
        out2 = expert2(x)
        out = self.alpha * out1 + (1 - self.alpha) * out2
        return out


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len
        self.indim = args.indim
        self.classes = args.classes

        # self.T = self.seq_len
        self.down_sampling_layers = args.down_sampling_layers
        self.temporalchannel = args.channel
        self.temporal_size = args.temporal_size
        self.feat_size = args.feat_size

        self.scale_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=1, out_channels=self.temporalchannel, kernel_size=(1, 3), padding=(0, 1))
                for i in range(self.down_sampling_layers + 1)
            ])

        self.temporalmapping = nn.ModuleList(
            [
                nn.Linear(self.temporal_size * self.indim, self.feat_size)
                for i in range(self.down_sampling_layers + 1)
            ])

        self.normalize_layers = torch.nn.ModuleList(
            [
                nn.BatchNorm2d(self.temporalchannel)
                for i in range(self.down_sampling_layers + 1)
            ])

        self.temporal_normalizes = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(self.feat_size)
                for i in range(self.down_sampling_layers + 1)
            ])

        self.temporal = torch.nn.ModuleList(
            [
                GRUmodel(self.temporalchannel, self.temporalchannel, self.temporal_size)
                for i in range(self.down_sampling_layers + 1)
            ])

        self.decompose = torch.nn.ModuleList(
            [
                nn.Linear(self.feat_size, self.feat_size)
                for i in range(self.down_sampling_layers + 1)
            ])

        self.attention = nn.Parameter((torch.ones(1, self.indim, 1)))

        self.classification = nn.Sequential(
            nn.Linear(self.feat_size * (self.down_sampling_layers + 1), self.classes)
        )

    def multi_scale_process(self, x, down_sampling_method='avg', down_sampling_window=2):
        """多尺度处理函数，动态适配设备"""
        if self.down_sampling_layers == 0:
            return [x]

        device = x.device
        down_pool = None

        if down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(down_sampling_window)
        elif down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.indim, out_channels=self.indim,
                                  kernel_size=3, padding=padding,
                                  stride=down_sampling_window,
                                  padding_mode='circular',
                                  bias=False).to(device)

        x_enc_ori = x
        temporal_list = []
        temporal_list.append(x.permute(0, 2, 1))

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            temporal_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        return temporal_list

    def sharp_softmax(self, logits, tau=1e-10):
        return F.softmax(logits / tau, dim=-1)

    def forward(self, x):
        device = x.device
        b, n, t = x.shape
        temporal_list = self.multi_scale_process(x)
        convs_list = []

        for i, x_scale in enumerate(temporal_list):
            x_scale = self.scale_convs[i](torch.unsqueeze(torch.transpose(x_scale, -1, -2), 1))
            x_scale = self.normalize_layers[i](x_scale)
            convs_list.append(x_scale.permute(0, 2, 3, 1).reshape(b * n, -1, self.temporalchannel))

        tem_list = []
        for i, x_conv in enumerate(convs_list):
            z = self.temporal[i](x_conv)[:, -1].reshape(b, n, -1) * self.attention
            tem_list.append(z)

        Embedding = []
        for i, z in enumerate(tem_list):
            z = self.temporalmapping[i](z.reshape(b, -1))
            z = self.temporal_normalizes[i](z)
            Embedding.append(z)

        Embedding = torch.stack(Embedding, dim=1).reshape(b, -1)
        out = self.classification(Embedding)
        return out
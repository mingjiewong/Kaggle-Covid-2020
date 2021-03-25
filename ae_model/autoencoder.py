import random
import os
import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pathlib import Path
from fastprogress import progress_bar
from data_processing.helpers import Config

random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)
torch.manual_seed(42)

class Conv1dStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dilation=1):
        super(Conv1dStack, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        h = self.res(x)
        return x + h


class Conv2dStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dilation=1):
        super(Conv2dStack, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        h = self.res(x)
        return x + h


class SeqEncoder(nn.Module):
    def __init__(self, config_path, in_dim: int):
        super(SeqEncoder, self).__init__()
        cfg = Config(config_path)
        self.conv0 = Conv1dStack(in_dim, cfg.units1, cfg.kernel_size1, padding=cfg.padding1)
        self.conv1 = Conv1dStack(cfg.units1, cfg.units2, cfg.kernel_size2, padding=cfg.padding2, dilation=cfg.dilation2)
        self.conv2 = Conv1dStack(cfg.units2, cfg.units3, cfg.kernel_size3, padding=cfg.padding3, dilation=cfg.dilation3)
        self.conv3 = Conv1dStack(cfg.units3, cfg.units4, cfg.kernel_size4, padding=cfg.padding4, dilation=cfg.dilation4)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


class BppAttn(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(BppAttn, self).__init__()
        self.conv0 = Conv1dStack(in_channel, out_channel, 3, padding=1)
        self.bpp_conv = Conv2dStack(5, out_channel)

    def forward(self, x, bpp):
        x = self.conv0(x)
        bpp = self.bpp_conv(bpp)
        x = torch.matmul(bpp, x.unsqueeze(-1))
        return x.squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerWrapper(nn.Module):
    def __init__(self, dmodel=256, nhead=8, num_layers=2):
        super(TransformerWrapper, self).__init__()
        self.pos_encoder = PositionalEncoding(256)
        encoder_layer = TransformerEncoderLayer(d_model=dmodel, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.pos_emb = PositionalEncoding(dmodel)

    def flatten_parameters(self):
        pass

    def forward(self, x):
        x = x.permute((1, 0, 2)).contiguous()
        x = self.pos_emb(x)
        x = self.transformer_encoder(x)
        x = x.permute((1, 0, 2)).contiguous()
        return x, None


class RnnLayers(nn.Module):
    def __init__(self, dmodel, dropout=0.3, transformer_layers: int = 2):
        super(RnnLayers, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.rnn0 = TransformerWrapper(dmodel, nhead=8, num_layers=transformer_layers)
        self.rnn1 = nn.LSTM(dmodel, dmodel // 2, batch_first=True, num_layers=1, bidirectional=True)
        self.rnn2 = nn.GRU(dmodel, dmodel // 2, batch_first=True, num_layers=1, bidirectional=True)

    def forward(self, x):
        self.rnn0.flatten_parameters()
        x, _ = self.rnn0(x)
        if self.rnn1 is not None:
            self.rnn1.flatten_parameters()
            x = self.dropout(x)
            x, _ = self.rnn1(x)
        if self.rnn2 is not None:
            self.rnn2.flatten_parameters()
            x = self.dropout(x)
            x, _ = self.rnn2(x)
        return x


class BaseAttnModel(nn.Module):
    def __init__(self, config_path='', transformer_layers: int = 2):
        super(BaseAttnModel, self).__init__()
        self.linear0 = nn.Linear(14 + 3, 1)
        self.seq_encoder_x = SeqEncoder(config_path, 18)
        self.attn = BppAttn(256, 128)
        self.seq_encoder_bpp = SeqEncoder(config_path, 128)
        self.seq = RnnLayers(256 * 2, dropout=0.3,
                             transformer_layers=transformer_layers)
        self.bpp_nb_mean = 0.077522 # mean of bpps_nb across all training data
        self.bpp_nb_std = 0.08914 # std of bpps_nb across all training data

    def get_bpp_feature(self, bpp):
        bpp_max = bpp.max(-1)[0]
        bpp_sum = bpp.sum(-1)
        bpp_nb = torch.true_divide((bpp > 0).sum(dim=1), bpp.shape[1])
        bpp_nb = torch.true_divide(bpp_nb - self.bpp_nb_mean, self.bpp_nb_std)
        return [bpp_max.unsqueeze(2), bpp_sum.unsqueeze(2), bpp_nb.unsqueeze(2)]

    def forward(self, x, bpp):
        bpp_features = self.get_bpp_feature(bpp[:, :, :, 0].float())
        x = torch.cat([x] + bpp_features, dim=-1)
        learned = self.linear0(x)
        x = torch.cat([x, learned], dim=-1)
        x = x.permute(0, 2, 1).contiguous().float()
        bpp = bpp.permute([0, 3, 1, 2]).contiguous().float()
        x = self.seq_encoder_x(x)
        bpp = self.attn(x, bpp)
        bpp = self.seq_encoder_bpp(bpp)
        x = x.permute(0, 2, 1).contiguous()
        bpp = bpp.permute(0, 2, 1).contiguous()
        x = torch.cat([x, bpp], dim=2)
        x = self.seq(x)
        return x


class AEModel(nn.Module):
    def __init__(self, config_path='', transformer_layers: int = 2):
        super(AEModel, self).__init__()
        self.seq = BaseAttnModel(config_path=config_path, transformer_layers=transformer_layers)
        self.linear = nn.Sequential(
            nn.Linear(256 * 2, 14),
            nn.Sigmoid(),
        )

    def forward(self, x, bpp):
        x = self.seq(x, bpp)
        x = F.dropout(x, p=0.3)
        x = self.linear(x)
        return x


class FromAeModel(nn.Module):
    def __init__(self, seq, pred_len=68, dmodel: int = 256):
        super(FromAeModel, self).__init__()
        self.seq = seq
        self.pred_len = pred_len
        self.linear = nn.Sequential(
            nn.Linear(dmodel * 2, 5),
        )

    def forward(self, x, bpp):
        x = self.seq(x, bpp)
        x = self.linear(x)
        x = x[:, :self.pred_len]
        return x

class TrainAE:
    def __init__(self, model):
        self.model = model
        self.lr_scheduler = None
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def learn_from_batch_ae(self, data, device):
        seq = data["sequence"].clone()
        seq[:, :, :14] = F.dropout2d(seq[:, :, :14], p=0.3)
        target = data["sequence"][:, :, :14]
        out = self.model(seq.to(device), data["bpp"].to(device))
        loss = F.binary_cross_entropy(out, target.to(device))

        return loss

    def train_ae(self, train_data, epochs=10, device="cpu", start_epoch: int = 0, start_it: int = 0,
                 MODEL_SAVE_PATH = './model'):
        print(f"device: {device}")
        losses = []
        it = start_it
        model_save_path = Path(MODEL_SAVE_PATH)
        start_epoch = start_epoch
        end_epoch = start_epoch + epochs
        min_loss = 10.0
        min_loss_epoch = 0
        if not model_save_path.exists():
            model_save_path.mkdir(parents=True)

        for epoch in progress_bar(range(start_epoch, end_epoch)):
            print(f"epoch: {epoch}")
            self.model.train()

            for i, data in enumerate(train_data):
                self.optimizer.zero_grad()
                loss = self.learn_from_batch_ae(data, device)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                loss_v = loss.item()
                losses.append(loss_v)
                it += 1
            loss_m = np.mean(losses)
            if loss_m < min_loss:
                min_loss_epoch = epoch
                min_loss = loss_m
            print(f'epoch: {epoch} loss: {loss_m}')
            losses = []
            torch.save(self.optimizer.state_dict(), str(model_save_path / "optimizer.pt"))
            torch.save(self.model.state_dict(), str(model_save_path / f"model-{epoch}.pt"))

        return dict(end_epoch=end_epoch, it=it, min_loss_epoch=min_loss_epoch)

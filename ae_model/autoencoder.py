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
        '''
        Load model parameters for 1D convolution stacks.

        Args:
          in_dim (int): dimension of input data
          out_dim (int): dimension of output data
          kernel_size (int): kernel size
          padding (int): padding size
          dilation (int): dilation size

        Attributes:
          conv (obj): first 1D convolution block
          res (obj): second 1D convolution residual block
        '''
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
        '''
        Run forward pass.

        Args:
          x (arr): input data

        Returns:
          arr: output data
        '''
        x = self.conv(x)
        h = self.res(x)
        return x + h


class Conv2dStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dilation=1):
        '''
        Load model parameters for 2D convolution stacks.

        Args:
          in_dim (int): dimension of input data
          out_dim (int): dimension of output data
          kernel_size (int): kernel size
          padding (int): padding size
          dilation (int): dilation size

        Attributes:
          conv (obj): first 2D convolution block
          res (obj): second 2D convolution residual block
        '''
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
        '''
        Run forward pass.

        Args:
          x (arr): input data

        Returns:
          arr: output data
        '''
        x = self.conv(x)
        h = self.res(x)
        return x + h


class SeqEncoder(nn.Module):
    def __init__(self, config_path, in_dim: int):
        '''
        Load model parameters for sequence encoder.

        Args:
          config_path (str): file path for config.yaml
          in_dim (int): dimension of input data

        Attributes:
          conv0 (obj): first stacked convolution block
          conv1 (obj): second stacked convolution block
          conv2 (obj): third stacked convolution block
          conv3 (obj): fourth stacked convolution block
        '''
        super(SeqEncoder, self).__init__()
        cfg = Config(config_path)
        self.conv0 = Conv1dStack(in_dim, cfg.units1, cfg.kernel_size1, padding=cfg.padding1)
        self.conv1 = Conv1dStack(cfg.units1, cfg.units2, cfg.kernel_size2, padding=cfg.padding2, dilation=cfg.dilation2)
        self.conv2 = Conv1dStack(cfg.units2, cfg.units3, cfg.kernel_size3, padding=cfg.padding3, dilation=cfg.dilation3)
        self.conv3 = Conv1dStack(cfg.units3, cfg.units4, cfg.kernel_size4, padding=cfg.padding4, dilation=cfg.dilation4)

    def forward(self, x):
        '''
        Run forward pass.

        Args:
          x (arr): input data

        Returns:
          arr: output data
        '''
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


class BppAttn(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        '''
        Load parameters for generating attention filter on sequence of base-pairing probabilities.

        Args:
          in_channel (int): dimension of input data
          out_channel (int): dimension of output data

        Attributes:
          conv0 (obj): 1D stacked convolution block
          bpp_conv (obj): 2D stacked convolution stacked block
        '''
        super(BppAttn, self).__init__()
        self.conv0 = Conv1dStack(in_channel, out_channel, 3, padding=1)
        self.bpp_conv = Conv2dStack(5, out_channel)

    def forward(self, x, bpp):
        '''
        Run forward pass.

        Args:
          x (arr): input data
          bpp (arr): base-pairing probabilities in a sequence with dimensions
            [len_sequence, len_sequence]

        Returns:
          arr: output data
        '''
        x = self.conv0(x)
        bpp = self.bpp_conv(bpp)
        x = torch.matmul(bpp, x.unsqueeze(-1))
        return x.squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        '''
        Load parameters for positional encoding.

        Args:
          d_model (int): number of latent dimensions
          dropout (int): dropout probability
          max_len (int): maximum vocabulary size

        Attributes:
          dropout (obj): dropout layer
          register_buffer (obj): register positional encoding matrix as a non-parameter
        '''
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
        '''
        Run forward pass.

        Args:
          x (arr): input data

        Returns:
          arr: output data
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerWrapper(nn.Module):
    def __init__(self, dmodel=256, nhead=8, num_layers=2):
        '''
        Load model parameters for both transformer and positional encoding blocks.

        Args:
          dmodel (int): number of latent dimensions
          nhead (int): number of attention heads
          num_layers (int): number of transformer layers

        Attributes:
          transformer_encoder (obj): transformer encoder block
          pos_emb (obj): positional encoder block
        '''
        super(TransformerWrapper, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=dmodel, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.pos_emb = PositionalEncoding(dmodel)

    def flatten_parameters(self):
        pass

    def forward(self, x):
        '''
        Run forward pass.

        Args:
          x (arr): input data

        Returns:
          arr: output data
          None
        '''
        x = x.permute((1, 0, 2)).contiguous()
        x = self.pos_emb(x)
        x = self.transformer_encoder(x)
        x = x.permute((1, 0, 2)).contiguous()
        return x, None


class RnnLayers(nn.Module):
    def __init__(self, dmodel, dropout=0.3, transformer_layers: int = 2):
        '''
        Load model parameters for RNN layers.

        Args:
          dmodel (int): number of latent dimensions
          dropout (int): dropout probability
          transformer_layers (int): number of transformer layers

        Attributes:
          dropout (obj): dropout layer
          rnn0 (obj): transformer block
          rnn1 (obj): LSTM block
          rnn2 (obj): GRU block
        '''
        super(RnnLayers, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.rnn0 = TransformerWrapper(dmodel, nhead=8, num_layers=transformer_layers)
        self.rnn1 = nn.LSTM(dmodel, dmodel // 2, batch_first=True, num_layers=1, bidirectional=True)
        self.rnn2 = nn.GRU(dmodel, dmodel // 2, batch_first=True, num_layers=1, bidirectional=True)

    def forward(self, x):
        '''
        Run forward pass.

        Args:
          x (arr): input data

        Returns:
          arr: output data
        '''
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
        '''
        Load parameters for base attention model.

        Args:
          config_path (str): file path for config.yaml
          transformer_layers (int): number of transformer layers

        Attributes:
          linear0 (obj): linear layer
          seq_encoder_x (obj): layer for sequence encoding on input data
          attn (obj): layer for generating attention filter on base-pairing probabilities
          seq_encoder_bpp (obj): layer for sequence encoding on base-pairing probabilities
          seq (obj): RNN layers
          bpp_nb_mean (float): mean number of non-zero base-pairing probabilities
          bpp_nb_std (float): standard deviation of number of non-zero base-pairing probabilities
        '''
        super(BaseAttnModel, self).__init__()
        self.linear0 = nn.Linear(14 + 3, 1)
        self.seq_encoder_x = SeqEncoder(config_path, 18)
        self.attn = BppAttn(256, 128)
        self.seq_encoder_bpp = SeqEncoder(config_path, 128)
        self.seq = RnnLayers(256 * 2, dropout=0.3,
                             transformer_layers=transformer_layers)
        self.bpp_nb_mean = 0.077522
        self.bpp_nb_std = 0.08914

    def get_bpp_feature(self, bpp):
        '''
        Generate general statistics of base-pairing probabilities in a sequence.

        Args:
          bpp (arr): base-pairing probabilities in a sequence with dimensions
            [len_sequence, len_sequence]

        Returns:
          arr: [maximum base-pairing probabilities,
            sum of base-pairing probabilities,
            normalized number of non-zero base-pairing probabilities]
        '''
        bpp_max = bpp.max(-1)[0]
        bpp_sum = bpp.sum(-1)
        bpp_nb = torch.true_divide((bpp > 0).sum(dim=1), bpp.shape[1])
        bpp_nb = torch.true_divide(bpp_nb - self.bpp_nb_mean, self.bpp_nb_std)
        return [bpp_max.unsqueeze(2), bpp_sum.unsqueeze(2), bpp_nb.unsqueeze(2)]

    def forward(self, x, bpp):
        '''
        Run forward pass.

        Args:
          x (arr): input data
          bpp (arr): base-pairing probabilities in a sequence with dimensions
            [len_sequence, len_sequence]

        Returns:
          arr: output data
        '''
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
        '''
        Load parameters for full model.

        Args:
          config_path (str): file path for config.yaml
          transformer_layers (int): number of transformer layers

        Attributes:
          seq (obj): base attention model
          linear (obj): linear layer with activation function
        '''
        super(AEModel, self).__init__()
        self.seq = BaseAttnModel(config_path=config_path, transformer_layers=transformer_layers)
        self.linear = nn.Sequential(
            nn.Linear(256 * 2, 14),
            nn.Sigmoid(),
        )

    def forward(self, x, bpp):
        '''
        Run forward pass.

        Args:
          x (arr): input data
          bpp (arr): base-pairing probabilities in a sequence with dimensions
            [len_sequence, len_sequence]

        Returns:
          arr: output data
        '''
        x = self.seq(x, bpp)
        x = F.dropout(x, p=0.3)
        x = self.linear(x)
        return x


class FromAeModel(nn.Module):
    def __init__(self, seq, pred_len=68, dmodel: int = 256):
        '''
        Load parameters for full model.

        Args:
          seq (obj): model
          pred_len (int): prediction length
          dmodel (int): number of latent dimensions

        Attributes:
          seq (obj): model
          pred_len (int): prediction length
          linear (obj): linear layer
        '''
        super(FromAeModel, self).__init__()
        self.seq = seq
        self.pred_len = pred_len
        self.linear = nn.Sequential(
            nn.Linear(dmodel * 2, 5),
        )

    def forward(self, x, bpp):
        '''
        Run forward pass.

        Args:
          x (arr): input data
          bpp (arr): base-pairing probabilities in a sequence with dimensions
            [len_sequence, len_sequence]

        Returns:
          arr: output data
        '''
        x = self.seq(x, bpp)
        x = self.linear(x)
        x = x[:, :self.pred_len]
        return x

class TrainAE:
    def __init__(self, model):
        '''
        Load parameters for full model, learning rate scheduler and optimizer.

        Args:
          model (obj): model

        Attributes:
          model (obj): model
          lr_scheduler (obj): learning rate scheduler
          optimizer (obj): optimizer
        '''
        self.model = model
        self.lr_scheduler = None
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def learn_from_batch_ae(self, data, device):
        '''
        Generate loss value.

        Args:
          data (arr): input data
          device (str): choice of gpu or cpu for running model

        Returns:
          loss (float): loss value
        '''
        seq = data["sequence"].clone()
        seq[:, :, :14] = F.dropout2d(seq[:, :, :14], p=0.3)
        target = data["sequence"][:, :, :14]
        out = self.model(seq.to(device), data["bpp"].to(device))
        loss = F.binary_cross_entropy(out, target.to(device))

        return loss

    def train_ae(self, train_data, epochs=10, device="cpu", start_epoch: int = 0, start_it: int = 0,
                 MODEL_SAVE_PATH = './model'):
        '''
        Run model.

        Args:
          train_data (arr): input data
          epochs (int): number of epochs
          device (str): choice of gpu or cpu for running model
          start_epoch (int): index of starting epoch
          start_it (int): index of starting sample from input data
          MODEL_SAVE_PATH (str): file path for saved model

        Returns:
          dict: index of ending epoch, index of ending sample from input data,
            index of epoch with minimum loss value
        '''
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

import torch
import random
import os
import numpy as np

from torch import nn
from fastprogress import progress_bar

random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)
torch.manual_seed(42)

class Loss:
    def __init__(self):
        pass

    def MCRMSE(self, y_true, y_pred):
        '''
        Generate mean columnwise root mean squared error.

        Args:
          y_true (arr): true target values
          y_pred (arr): predicted target values

        Returns:
          float: mean columnwise root mean squared error
        '''
        colwise_mse = torch.mean(torch.square(y_true - y_pred), dim=1)
        return torch.mean(torch.sqrt(colwise_mse), dim=1)

    def sn_mcrmse_loss(self, predict, target, signal_to_noise):
        '''
        Generate signal-to-noise scaled mean columnwise root mean squared error.

        Args:
          predict (arr): true target values
          target (arr): predicted target values
          signal_to_noise (float): mean measurement value over mean statistical error in measurement value

        Returns:
          float: signal-to-noise scaled mean columnwise root mean squared error
        '''
        loss = self.MCRMSE(target, predict)
        weight = 0.5 * torch.log(signal_to_noise + 1.01)
        loss = (loss * weight).mean()
        return loss

    def learn_from_batch(self, model, data, optimizer, lr_scheduler, device):
        '''
        Run training on batch.

        Args:
          model (obj): model
          data (arr): training data
          optimizer (obj): optimizer
          lr_scheduler (obj): learning rate scheduler
          device (str): choice of gpu or cpu for running model

        Returns:
          arr: predicted values
          float: loss value
        '''
        optimizer.zero_grad()
        out = model(data["sequence"].to(device), data["bpp"].to(device))
        signal_to_noise = data["signal_to_noise"] * data["score"]
        loss = self.sn_mcrmse_loss(out, data["label"].to(device), signal_to_noise.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        return out, loss

    def evaluate(self, model, valid_data, device):
        '''
        Run validation.

        Args:
          model (obj): model
          valid_data (arr): validation data
          device (str): choice of gpu or cpu for running model

        Returns:
          dict: mean loss value, mean of mean columnwise root mean squared error
        '''
        model.eval()
        loss_list = []
        mcrmse = []
        for i, data in enumerate(valid_data):
            with torch.no_grad():
                y = model(data["sequence"].to(device), data["bpp"].to(device))
                mcrmse_ = self.MCRMSE(data["label"].to(device), y)[data["signal_to_noise"] > 1]
                mcrmse.append(mcrmse_.mean().item())
                loss = self.sn_mcrmse_loss(y, data["label"].to(device), data["signal_to_noise"].to(device))
                loss_list.append(loss.item())
        model.train()

        return dict(loss=np.mean(loss_list), mcmse=np.mean(mcrmse))

class Predict:
    def __init__(self):
        pass

    def predict_batch(self, model, data, device, target_cols):
        '''
        Run prediction on batch.

        Args:
          model (obj): model
          data (arr): test data
          device (str): choice of gpu or cpu for running model
          target_cols (arr): target features

        Returns:
          arr: list of dict of target features and their corresponding predictions
        '''
        with torch.no_grad():
            pred = model(data["sequence"].to(device), data["bpp"].to(device))
            pred = pred.detach().cpu().numpy()
        return_values = []
        ids = data["ids"]
        for idx, p in enumerate(pred):
            id_ = ids[idx]
            assert p.shape == (model.pred_len, len(target_cols))
            for seqpos, val in enumerate(p):
                assert len(val) == len(target_cols)
                dic = {key: val for key, val in zip(target_cols, val)}
                dic["id_seqpos"] = f"{id_}_{seqpos}"
                return_values.append(dic)
        return return_values

    def predict_data(self, model, loader, device, batch_size, target_cols):
        '''
        Run prediction.

        Args:
          model (obj): model
          loader (arr): test data
          device (str): choice of gpu or cpu for running model
          batch_size (int): batch size
          target_cols (arr): target features

        Returns:
          arr: list of batches of dict of target features and their corresponding predictions
        '''
        data_list = []
        for i, data in enumerate(progress_bar(loader)):
            data_list += self.predict_batch(model, data, device, target_cols)
        expected_length = model.pred_len * len(loader) * batch_size
        assert len(data_list) == expected_length, f"len = {len(data_list)} expected = {expected_length}"
        return data_list

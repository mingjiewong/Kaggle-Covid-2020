import torch
import random
import os
import numpy as np
from sklearn.model_selection import ShuffleSplit

random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)
torch.manual_seed(42)

class Config(object):
    def __init__(self):
        self.BATCH_SIZE = 64
        self.k_folds = 5 #12
        self.lr_scheduler = None
        self.test_size = .1
        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        self.epoch = 10 #200

class Loss:
    def __init__(self):
        pass

    def MCRMSE(self, y_true, y_pred):
        colwise_mse = torch.mean(torch.square(y_true - y_pred), dim=1)
        return torch.mean(torch.sqrt(colwise_mse), dim=1)

    def sn_mcrmse_loss(self, predict, target, signal_to_noise):
        loss = self.MCRMSE(target, predict)
        weight = 0.5 * torch.log(signal_to_noise + 1.01)
        loss = (loss * weight).mean()
        return loss

    def learn_from_batch(self, model, data, optimizer, lr_scheduler, device):
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
        # batch x seq_len x target_size
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
        data_list = []
        for i, data in enumerate(progress_bar(loader)):
            data_list += self.predict_batch(model, data, device, target_cols)
        expected_length = model.pred_len * len(loader) * batch_size
        assert len(data_list) == expected_length, f"len = {len(data_list)} expected = {expected_length}"
        return data_list

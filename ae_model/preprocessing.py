import random
import os
import torch
import functools
import pandas as pd
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset

random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)
torch.manual_seed(42)

class Load:
    def __init__(self, base_train_data='', base_test_data=''):
        self.base_train_data = pd.read_json(base_train_data, lines=True)
        self.base_test_data = pd.read_json(base_test_data, lines=True)
        self.target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
        self.input_cols = ['sequence', 'structure', 'predicted_loop_type']
        self.error_cols = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_Mg_50C', 'deg_error_pH10', 'deg_error_50C']
        self.token_dicts = {"sequence": {x: i for i, x in enumerate("ACGU")},
                            "structure": {x: i for i, x in enumerate('().')},
                            "predicted_loop_type": {x: i for i, x in enumerate("BEHIMSX")}}

    def denoise(self):
        denoised_train_data = self.base_train_data[self.base_train_data.signal_to_noise > 1].reset_index(drop = True)

        return denoised_train_data

    def query_seq_length(self):
        public_df = self.base_test_data.query("seq_length == 107").copy()
        private_df = self.base_test_data.query("seq_length == 130").copy()
        public_df = public_df.reset_index()
        private_df = private_df.reset_index()

        return public_df, private_df

    def preprocess_feature_col(self, loaded_data, col):
        dic = self.token_dicts[col]
        dic_len = len(dic)
        seq_length = len(loaded_data[col][0])
        ident = np.identity(dic_len)
        # convert to one hot
        arr = np.array(
            loaded_data[[col]].applymap(lambda seq: [ident[dic[x]] for x in seq]).values.tolist()
        ).squeeze(1)
        # shape: data_size x seq_length x dic_length
        assert arr.shape == (len(loaded_data), seq_length, dic_len)
        return arr

    def preprocess_inputs(self, loaded_data, cols):
        return np.concatenate([self.preprocess_feature_col(loaded_data, col) for col in cols], axis=2)

    def preprocess(self, loaded_data, is_test=False):
        inputs = self.preprocess_inputs(loaded_data, self.input_cols)
        if is_test:
            labels = None
        else:
            labels = np.array(loaded_data[self.target_cols].values.tolist()).transpose((0, 2, 1))
            assert labels.shape[2] == len(self.target_cols)

        assert inputs.shape[2] == 14
        return inputs, labels


class VacDataset(Dataset):
    def __init__(self, features, loaded_data, structure_adj, distance_matrix, path_bpps, labels=None):
        self.features = features
        self.labels = labels
        self.test = labels is None
        self.ids = loaded_data["id"]
        self.score = None
        self.structure_adj = structure_adj
        self.distance_matrix = distance_matrix
        self.path_bpps = path_bpps
        if "score" in loaded_data.columns:
            self.score = loaded_data["score"]
        else:
            loaded_data["score"] = 1.0
            self.score = loaded_data["score"]
        self.signal_to_noise = None
        if not self.test:
            self.signal_to_noise = loaded_data["signal_to_noise"]
            assert self.features.shape[0] == self.labels.shape[0]
        else:
            assert self.ids is not None

    def __len__(self):
        return len(self.features)

    @functools.lru_cache(5000)
    def load_from_id_data(self, id_):
        path = Path(self.path_bpps+f"{id_}.npy")
        data = np.load(str(path))
        return data

    def __getitem__(self, index):
        bpp = torch.from_numpy(self.load_from_id_data(self.ids[index]).copy()).float()
        adj = self.structure_adj[index]
        distance = self.distance_matrix[0]
        bpp = np.concatenate([bpp[:, :, None], adj, distance], axis=2)
        if self.test:
            return dict(sequence=self.features[index].float(), bpp=bpp, ids=self.ids[index])
        else:
            return dict(sequence=self.features[index].float(), bpp=bpp,
                        label=self.labels[index], ids=self.ids[index],
                        signal_to_noise=self.signal_to_noise[index],
                        score=self.score[index])

class CreateLoader:
    def __init__(self):
        pass

    def get_distance_matrix(self, leng):
        idx = np.arange(leng)
        Ds = []
        for i in range(len(idx)):
            d = np.abs(idx[i] - idx)
            Ds.append(d)

        Ds = np.array(Ds) + 1
        Ds = 1 / Ds
        Ds = Ds[None, :, :]
        Ds = np.repeat(Ds, 1, axis=0)

        Dss = []
        for i in [1, 2, 4]:
            Dss.append(Ds ** i)
        Ds = np.stack(Dss, axis=3)
        return Ds

    def get_structure_adj(self, loaded_data):
        Ss = []
        for i in range(len(loaded_data)):
            seq_length = loaded_data["seq_length"].iloc[i]
            structure = loaded_data["structure"].iloc[i]
            sequence = loaded_data["sequence"].iloc[i]

            cue = []
            a_structures = OrderedDict([
                (("A", "U"), np.zeros([seq_length, seq_length])),
                (("C", "G"), np.zeros([seq_length, seq_length])),
                (("U", "G"), np.zeros([seq_length, seq_length])),
                (("U", "A"), np.zeros([seq_length, seq_length])),
                (("G", "C"), np.zeros([seq_length, seq_length])),
                (("G", "U"), np.zeros([seq_length, seq_length])),
            ])
            for j in range(seq_length):
                if structure[j] == "(":
                    cue.append(j)
                elif structure[j] == ")":
                    start = cue.pop()
                    a_structures[(sequence[start], sequence[j])][start, j] = 1
                    a_structures[(sequence[j], sequence[start])][j, start] = 1

            a_strc = np.stack([a for a in a_structures.values()], axis=2)
            a_strc = np.sum(a_strc, axis=2, keepdims=True)
            Ss.append(a_strc)

        Ss = np.array(Ss)
        return Ss

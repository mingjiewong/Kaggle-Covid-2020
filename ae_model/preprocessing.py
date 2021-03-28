import random
import os
import torch
import functools
import pandas as pd
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset
from pathlib import Path

random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)
torch.manual_seed(42)

class Load:
    def __init__(self, base_train_data='', base_test_data=''):
        '''
        Read json files for features of train and test data respectively.

        Args:
          base_train_data (str): file path of features for train data
          base_test_data (str): file path of features for test data

        Attributes:
          base_train_data (dataframe): input data of features for train data
          base_test_data (dataframe): input data of features for test data
          target_cols (arr): list of target features
          input_cols (arr): list of train (sequential) features
          token_dicts (dict): dict of train (sequential) features and their corresponding tokens
        '''
        self.base_train_data = pd.read_json(base_train_data, lines=True)
        self.base_test_data = pd.read_json(base_test_data, lines=True)
        self.target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
        self.input_cols = ['sequence', 'structure', 'predicted_loop_type']
        self.token_dicts = {"sequence": {x: i for i, x in enumerate("ACGU")},
                            "structure": {x: i for i, x in enumerate('().')},
                            "predicted_loop_type": {x: i for i, x in enumerate("BEHIMSX")}}

    def denoise(self):
        '''
        Drop samples with noise from training data.

        Returns:
          dataframe: denoised input data of features for train data
        '''
        denoised_train_data = self.base_train_data[self.base_train_data.signal_to_noise > 1].reset_index(drop = True)

        return denoised_train_data

    def query_seq_length(self):
        '''
        Generate test data with sequence length of 107 and 130 only respectively.

        Returns:
          dataframe: updated input data of features for test data with sequence length of 107 only
          dataframe: updated input data of features for test data with sequence length of 130 only
        '''
        public_df = self.base_test_data.query("seq_length == 107").copy()
        private_df = self.base_test_data.query("seq_length == 130").copy()
        public_df = public_df.reset_index()
        private_df = private_df.reset_index()

        return public_df, private_df

    def preprocess_feature_col(self, loaded_data, col):
        '''
        One hot encode a sequential feature of the input data based on the presence of each possible token of the feature at each position of the sequence.

        Args:
          loaded_data (dataframe): input data
          col (str): sequential feature

        Returns:
          arr: updated input data of only one-hot encoded features with dimensions
            [number of samples, sequence length, number of tokens]
        '''
        dic = self.token_dicts[col]
        dic_len = len(dic)
        seq_length = len(loaded_data[col][0])
        ident = np.identity(dic_len)
        arr = np.array(
            loaded_data[[col]].applymap(lambda seq: [ident[dic[x]] for x in seq]).values.tolist()
        ).squeeze(1)
        assert arr.shape == (len(loaded_data), seq_length, dic_len)
        return arr

    def preprocess_inputs(self, loaded_data, cols):
        '''
        Apply one hot encoding on a list of sequential features of the input data.

        Args:
          loaded_data (dataframe): input data
          cols (arr): list of sequential features

        Returns:
          arr: updated input data of only one-hot encoded features with dimensions
            [number of samples, sequence length, total number of tokens across all features]
        '''
        return np.concatenate([self.preprocess_feature_col(loaded_data, col) for col in cols], axis=2)

    def preprocess(self, loaded_data, is_test=False):
        '''
        Run preprocessing.

        Args:
          loaded_data (dataframe): input data
          is_test (bool): whether the input data is train or test

        Returns:
          array: updated input data of only one-hot encoded train (sequential) features with dimensions
            [number of samples, sequence length, 14]
          array: updated input data of only target features with dimensions
            [number of samples, sequence length, 5]
        '''
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
        '''
        Load all feature values from input data.

        Args:
          features (array): input data of only one-hot encoded train (sequential) features
          loaded_data (dataframe): input data
          structure_adj (arr): adjacency matrices
          distance_matrix (arr): distance matrix
          path_bpps (str): directory path for sample files of base-pairing probabilities
          labels (array): input data of only target features

        Attributes:
          features (array): input data of only one-hot encoded train (sequential) features
          labels (array): input data of only target features
          test (bool): whether the input data is train or test
          ids (int): sample id
          score (int): number of base positions used in scoring of target values
          structure_adj (arr): adjacency matrices
          distance_matrix (arr): distance matrix
          path_bpps (str): directory path for sample files of base-pairing probabilities
          signal_to_noise (float): mean measurement value over mean statistical error in measurement value
        '''
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
        '''
        Calculate number of features.

        Returns:
          int: number of features
        '''
        return len(self.features)

    @functools.lru_cache(5000)
    def load_from_id_data(self, id_):
        '''
        Load sequence of base-pairing probabilities of a sample.

        Args:
          id_ (int): sample id

        Returns:
          arr: sequence of base-pairing probabilities
        '''
        path = Path(self.path_bpps+f"{id_}.npy")
        data = np.load(str(path))
        return data

    def __getitem__(self, index):
        '''
        Load the dict of feature values of a sample.

        Args:
          index (int): sample id

        Return:
          dict: dict of feature values
        '''
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
        '''
        Generate distance matrix from the input data.

        Args:
          leng (int): sequence length

        Returns:
          arr: distance matrix with dimensions
            [1, sequence length, sequence length, 3]
        '''
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
        '''
        Generate adjacency matrices from the sequence of each sample in the input data.

        Args:
          loaded_data (dataframe): input data

        Returns:
          arr: list of adjacency matrices with dimensions
            [number of samples, sequence length, sequence length, 1]
        '''
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

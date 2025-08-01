import torch
from preprocess.parse_trees import ZincGrammarModel
import numpy as np
import pandas as pd
import sys
import os
from contextlib import contextmanager
# 创建模型
model = ZincGrammarModel()
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class MolGraphDataset():
    def __init__(self, path):
        print(path)

        file = pd.read_csv(path, sep=',')

        n_cols = file.shape[1]
        self.header_cols = np.genfromtxt(path, delimiter=',', usecols=range(0, n_cols), dtype=np.str_, comments=None)
        self.target_names = [self.header_cols[1]]
        self.smiles_1 = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=[0], dtype=np.str_, comments=None)
        self.targets_1 = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=[1], dtype=np.int_, comments=None)
        self.valid_indices = [i for i in range(len(self.smiles_1)) if self.is_valid(i)]
        self.smiles = [self.smiles_1[i] for i in self.valid_indices]
        self.targets = [self.targets_1[i] for i in self.valid_indices]


    def __getitem__(self, index):
        smiles = self.smiles[index]
        target = self.targets[index]
        token_seq = model.encode_smiles(smiles)
        tokens_idx, atom_mask_list = construct_input_from_tokenseq(token_seq, max_len=500)
        tokens_idx = np.array(tokens_idx)
        atom_mask = np.array(atom_mask_list)
        target = np.array(target)

        # print(tokens_idx)

        return tokens_idx, atom_mask, target

    def is_valid(self, index):
        try:
            smiles = self.smiles_1[index]
            token_seq = model.encode_smiles(smiles)
            construct_input_from_tokenseq(token_seq, max_len=500)

            return True
        except:
            # 数据无效
            return False

    def __len__(self):
        return len(self.valid_indices)



def molgraph_collate_fn(data):
    batch_size = len(data)
    tokens_idx_batch = torch.zeros(batch_size, 501, dtype=torch.long)
    atom_mask_batch = torch.zeros(batch_size, 501, dtype=torch.long)
    target_num = 1
    target_batch = torch.zeros(batch_size, target_num)

    for i in range(batch_size):
        tokens_idx, atom_mask, target = data[i]
        tokens_idx = torch.tensor(tokens_idx,dtype=torch.long)
        atom_mask = torch.tensor(atom_mask, dtype=torch.long)
        target = torch.tensor(target)

        tokens_idx_batch[i,:] = tokens_idx
        atom_mask_batch[i,:] = atom_mask
        target_batch[i, :] = target

    return tokens_idx_batch, atom_mask_batch, target_batch

def construct_input_from_tokenseq(token_list, max_len=1000):
    grammer_token_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                      20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                      41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                      62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
    all_token_list = ['[PAD]', '[GLO]', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                      20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                      41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                      62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]
    word2idx = {'[PAD]':0, '[GLO]':82}

    if len(token_list) > max_len:
        token_list = token_list[:max_len]

    padding_list = ['[PAD]'] * (max_len - len(token_list))
    tokens = ['[GLO]'] + token_list + padding_list

    tokens_idx = []
    atom_mask_list = []

    #81和0要被mask掉
    for token in tokens:
        if token in grammer_token_list:
            atom_mask_list.append(1)
            tokens_idx.append(token)
        elif token=='[GLO]':
            atom_mask_list.append(1)
            tokens_idx.append(word2idx['[GLO]'])
        else:
            atom_mask_list.append(0)
            tokens_idx.append(0)

    return tokens_idx, atom_mask_list


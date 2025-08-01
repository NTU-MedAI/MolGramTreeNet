import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit import RDLogger
import sys
import os
from contextlib import contextmanager
from preprocess.parse_trees import ZincGrammarModel

RDLogger.DisableLog('rdApp.*')
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


ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


class MolGraphDataset():
    def __init__(self, path):
        file = pd.read_csv(path, sep=',')

        n_cols = file.shape[1]
        self.header_cols = np.genfromtxt(path, delimiter=',', usecols=range(0, n_cols), dtype=np.str_, comments=None)
        self.target_names = self.header_cols[1:3]  # Two labels: FDA_APPROVED and CT_TOX
        self.smiles_1 = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=[0], dtype=np.str_, comments=None)
        self.targets_1 = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=[1, 2], dtype=np.int_, comments=None)
        self.valid_indices = [i for i in range(len(self.smiles_1)) if self.is_valid(i)]
        self.smiles = [self.smiles_1[i] for i in self.valid_indices]
        self.targets = [self.targets_1[i] for i in self.valid_indices]

    def __getitem__(self, index):
        smiles = self.smiles[index]
        target = self.targets[index]

        with suppress_output():
            token_seq = model.encode_smiles(smiles)

        tokens_idx, atom_mask_list = construct_input_from_tokenseq(token_seq, max_len=500)
        tokens_idx = np.array(tokens_idx)
        atom_mask = np.array(atom_mask_list)
        target = np.array(target)

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        type_idx = []
        chirality_idx = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        return tokens_idx, atom_mask, target, x, edge_index, edge_attr

    def is_valid(self, index):
        try:
            smiles = self.smiles_1[index]
            with suppress_output():
                token_seq = model.encode_smiles(smiles)
                construct_input_from_tokenseq(token_seq, max_len=500)

            Chem.MolFromSmiles(smiles)
            return True
        except:
            return False

    def __len__(self):
        return len(self.valid_indices)


def molgraph_collate_fn(data):
    batch_size = len(data)
    tokens_idx_batch = torch.zeros(batch_size, 501, dtype=torch.long)
    atom_mask_batch = torch.zeros(batch_size, 501, dtype=torch.long)
    target_num = 2
    target_batch = torch.zeros(batch_size, target_num)

    x_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_indices = []
    num_nodes = 0

    for i in range(batch_size):
        tokens_idx, atom_mask, target, x, edge_index, edge_attr = data[i]
        tokens_idx = torch.tensor(tokens_idx, dtype=torch.long)
        atom_mask = torch.tensor(atom_mask, dtype=torch.long)
        target = torch.tensor(target)

        tokens_idx_batch[i, :] = tokens_idx
        atom_mask_batch[i, :] = atom_mask
        target_batch[i, :] = target

        x_list.append(x)
        edge_index_list.append(edge_index + num_nodes)
        edge_attr_list.append(edge_attr)
        batch_indices.append(torch.full((x.size(0),), i, dtype=torch.long))
        num_nodes += x.size(0)

    x_batch = torch.cat(x_list, dim=0)
    edge_index_batch = torch.cat(edge_index_list, dim=1)
    edge_attr_batch = torch.cat(edge_attr_list, dim=0)
    batch_batch = torch.cat(batch_indices, dim=0)

    return tokens_idx_batch, atom_mask_batch, target_batch, x_batch, edge_index_batch, edge_attr_batch, batch_batch


def construct_input_from_tokenseq(token_list, max_len=1000):
    grammer_token_list = list(range(1, 81))
    all_token_list = ['[PAD]', '[GLO]'] + grammer_token_list
    word2idx = {'[PAD]': 0, '[GLO]': 82}

    if len(token_list) > max_len:
        token_list = token_list[:max_len]

    padding_list = ['[PAD]'] * (max_len - len(token_list))
    tokens = ['[GLO]'] + token_list + padding_list

    tokens_idx = []
    atom_mask_list = []

    for token in tokens:
        if token in grammer_token_list:
            atom_mask_list.append(1)
            tokens_idx.append(token)
        elif token == '[GLO]':
            atom_mask_list.append(1)
            tokens_idx.append(word2idx['[GLO]'])
        else:
            atom_mask_list.append(0)
            tokens_idx.append(0)

    return tokens_idx, atom_mask_list
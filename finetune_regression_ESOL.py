import numpy as np
import os
import copy
from time import time, strftime, localtime
import torch
import torch.optim as opt
from utils import parse_args, Logger
from torch import nn
from dataloader.data_process_ESOL import MolGraphDataset, molgraph_collate_fn, suppress_output
from torch.utils.data import DataLoader, random_split, Subset
from model.my_nn import BERT_atom_embedding_generator
from torch.nn.functional import mse_loss
from model.gcn_finetune import GCN
from model.my_fusion_model import CombinedModel


class FullModel(nn.Module):
    def __init__(self, combined_model, predictor):
        super().__init__()
        self.combined_model = combined_model
        self.predictor = predictor

    def forward(self, token_idx, x, edge_index, edge_attr, batch):
        combined_out = self.combined_model(token_idx, x, edge_index, edge_attr, batch)
        return self.predictor(combined_out)


class predictor(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(dim, dim * 2), nn.Dropout(p=dropout), nn.ReLU(), nn.Linear(dim * 2, 1))

    def forward(self, feats):
        return self.out(feats)


def run_eval(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_pred = []
    y_val = []
    with torch.no_grad():
        batch_idx_val = 0
        for token_idx, atom_mask, target, x, edge_index, edge_attr, batch in loader:
            token_idx, atom_mask = token_idx.to(device), atom_mask.to(device)
            target = target.to(device)
            x, edge_index, edge_attr, batch = (x.to(device),
                                               edge_index.to(device),
                                               edge_attr.to(device),
                                               batch.to(device))

            batch_idx_val += 1

            out = model(token_idx, x, edge_index, edge_attr, batch)

            if torch.isnan(out).any():
                continue

            loss = mse_loss(out, target, reduction='sum').item()
            total_loss += loss
            total_samples += target.numel()

            y_val.append(target.detach().cpu())
            y_pred.append(out.detach().cpu())

    mean_squared_error = total_loss / total_samples
    rmse = np.sqrt(mean_squared_error)

    return rmse


def main():
    args = parse_args()
    args.save_path = 'save/'
    torch.manual_seed(2024)
    log = Logger(f'{args.save_path}ESOL/', f'ESOL_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')
    args.epochs = 100
    args.gpu = '0'
    args.lr = 2e-5 * len(args.gpu.split(','))
    args.bs = 16 * len(args.gpu.split(','))
    args.data = 'measured log solubility in mols per litre'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with suppress_output():
        full_dataset = MolGraphDataset('dataset/delaney-processed.csv')

    length = len(full_dataset)
    indices = list(range(length))

    train = int(length * 0.8)
    test_guodu = length - train
    if test_guodu % args.bs != 0:
        test_guodu = (test_guodu // args.bs) * args.bs
        train = length - test_guodu
        if train % args.bs == 1:
            train = train + 2
            test_guodu = test_guodu - 2
    val = int(test_guodu / 2)
    test = test_guodu - val

    train_indices, val_indices, test_indices = random_split(indices, lengths=[train, val, test])

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=molgraph_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, collate_fn=molgraph_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, collate_fn=molgraph_collate_fn)

    d_model = 768
    feat_dim = 300
    one_model = BERT_atom_embedding_generator(d_model=d_model, n_layers=6, vocab_size=83,
                                              maxlen=501, d_k=64, d_v=64, n_heads=12, d_ff=768 * 4,
                                              global_label_dim=1, atom_label_dim=15)
    two_model = GCN(num_layer=5, emb_dim=300, feat_dim=feat_dim, drop_ratio=0.1, pool='mean')

    combined_model = CombinedModel(one_model, two_model)
    Predictor = predictor(d_model + feat_dim)

    model = FullModel(combined_model, Predictor)
    model.to(device)

    if len(args.gpu) > 1:  model = torch.nn.DataParallel(model)

    best_metric = 1e9
    criterion = torch.nn.MSELoss()
    optimizer = opt.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=10, min_lr=5e-6)
    log.logger.info(f'{"=" * 60} ESOL {"=" * 60}\n'
                    f'Train: {len(train_dataset)}; Val: {len(val_dataset)}; Test: {len(test_dataset)}'
                    f'\nTarget: {args.data}; Batch_size: {args.bs}\nStart Training {"=" * 60}')

    t0 = time()
    early_stop = 0
    RMSE_list = []

    for epoch in range(0, args.epochs):
        model.train()
        loss = 0.0
        t1 = time()
        batch_idx = 0
        for token_idx, atom_mask, target, x, edge_index, edge_attr, batch in train_loader:
            token_idx, atom_mask = token_idx.to(device), atom_mask.to(device)
            x, edge_index, edge_attr, batch = (x.to(device),
                                               edge_index.to(device),
                                               edge_attr.to(device),
                                               batch.to(device))
            target = target.to(device)

            batch_idx += 1
            optimizer.zero_grad()

            out = model(token_idx, x, edge_index, edge_attr, batch)

            if torch.isnan(out).any():
                continue
            loss_batch = criterion(out, target.float())

            loss += loss_batch.item() / (len(target) * args.bs)

            loss_batch.backward()
            optimizer.step()

        metric = run_eval(model, val_loader, device)

        metric_test = run_eval(model, test_loader, device)
        RMSE_list.append(metric_test)

        log.logger.info(
            'Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | val_RMSE: {:.3f} | test_RMSE: {:.3f} | '
            '| Lr: {:.3f}'.format(epoch + 1, time() - t1, loss * 1e4, metric, metric_test,
                                  optimizer.param_groups[0]['lr'] * 1e5))
        lr_scheduler.step(metric)

        if metric < best_metric:
            best_metric = metric
            best_model = copy.deepcopy(model)
            best_epoch = epoch + 1
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= 50: log.logger.info('Early Stopping!!! No Improvement on Loss for 100 Epochs.'); break

    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    checkpoint = {'epochs': args.epochs}

    best_test_RMSE = min(RMSE_list)
    best_epoch = RMSE_list.index(best_test_RMSE) + 1

    if len(args.gpu) > 1:
        checkpoint['model'] = best_model.module.state_dict()
    else:
        checkpoint['model'] = best_model.state_dict()
    torch.save(checkpoint, args.save_path + f'ESOL_model.pt')
    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    log.logger.info(f'Save the best model as ESOL_model.pt.\n')
    log.logger.info('Best Epoch: {} | test_RMSE: {:.5f}'.format(best_epoch, best_test_RMSE))


if __name__ == '__main__':
    main()
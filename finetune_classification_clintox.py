import os
import copy
from time import time, strftime, localtime
import torch
import torch.optim as opt
from sklearn.metrics import roc_auc_score
from utils import parse_args, Logger, set_seed
from torch import nn
from dataloader.data_process_clintox import MolGraphDataset, molgraph_collate_fn, suppress_output
from torch.utils.data import DataLoader, random_split, Subset
from model.my_nn import BERT_atom_embedding_generator
from model.gcn_finetune import GCN
from model.my_fusion_model import CombinedModel
import numpy as np


class FullModel(nn.Module):
    def __init__(self, combined_model, predictor):
        super().__init__()
        self.combined_model = combined_model
        self.predictor = predictor

    def forward(self, token_idx, x, edge_index, edge_attr, batch):
        combined_out = self.combined_model(token_idx, x, edge_index, edge_attr, batch)
        return self.predictor(combined_out)


class Predictor(nn.Module):
    def __init__(self, dim, num_labels, dropout=0.1):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(dim * 2, num_labels)
        )

    def forward(self, feats):
        return self.out(feats)


def run_eval(args, model, loader, device):
    model.eval()
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
            out = torch.sigmoid(out)

            if torch.isnan(out).any():
                continue

            y_pred.append(out)
            y_val.append(target)
    y_pred = torch.cat(y_pred)
    y_val = torch.cat(y_val)

    aurocs = []
    for i in range(y_val.shape[1]):
        try:
            auroc = roc_auc_score(y_val[:, i].cpu().numpy(), y_pred[:, i].cpu().numpy())
            aurocs.append(auroc)
        except ValueError:
            pass

    mean_auroc = np.mean(aurocs)
    return mean_auroc


def main():
    args = parse_args()
    args.save_path = 'save/'
    torch.manual_seed(2000)
    log = Logger(f'{args.save_path}clintox/', f'clintox_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')
    args.epochs = 100
    args.gpu = '0'
    args.lr = 2e-5 * len(args.gpu.split(','))
    args.bs = 16 * len(args.gpu.split(','))
    args.data = 'clintox'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with suppress_output():
        full_dataset = MolGraphDataset('/home/ntu/Documents/zyk/MolGramTreeNet/data/ClinTox/raw/clintox.csv')

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
    num_labels = 2

    one_model = BERT_atom_embedding_generator(d_model=d_model, n_layers=6, vocab_size=83,
                                              maxlen=501, d_k=64, d_v=64, n_heads=12, d_ff=768 * 4,
                                              global_label_dim=1, atom_label_dim=15)
    two_model = GCN(num_layer=5, emb_dim=300, feat_dim=feat_dim, drop_ratio=0.1, pool='mean')

    combined_model = CombinedModel(one_model, two_model)
    predictor_model = Predictor(d_model + feat_dim, num_labels)

    model = FullModel(combined_model, predictor_model)
    model.to(device)

    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model)

    best_metric = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = opt.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.6, patience=10, min_lr=5e-6)
    log.logger.info(f'{"=" * 60} clintox {"=" * 60}\n'
                    f'Train: {len(train_dataset)}; Val: {len(val_dataset)}; Test: {len(test_dataset)}'
                    f'\nTarget: {args.data}; Batch_size: {args.bs}\nStart Training {"=" * 60}')

    t0 = time()
    early_stop = 0
    auc_list = []

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

        metric = run_eval(args, model, val_loader, device)
        metric_test = run_eval(args, model, test_loader, device)
        auc_list.append(metric_test)

        log.logger.info(
            'Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | val_ROC-AUC: {:.5f} | test_ROC-AUC: {:.5f}'
            '| Lr: {:.3f}'.format(epoch + 1, time() - t1, loss * 1e4, metric, metric_test,
                                  optimizer.param_groups[0]['lr'] * 1e5))
        lr_scheduler.step(metric)

        if metric > best_metric:
            best_metric = metric
            best_model = copy.deepcopy(model)
            best_epoch = epoch + 1
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= 50:
            log.logger.info('Early Stopping!!! No Improvement on Loss for 50 Epochs.')
            break

    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    checkpoint = {'epochs': args.epochs}

    best_test_auc = max(auc_list)
    best_epoch = auc_list.index(best_test_auc) + 1

    if len(args.gpu) > 1:
        checkpoint['model'] = best_model.module.state_dict()
    else:
        checkpoint['model'] = best_model.state_dict()

    if not os.path.exists(args.save_path + 'clintox/'):
        os.makedirs(args.save_path + 'clintox/')
    torch.save(checkpoint, args.save_path + f'clintox/clintox_model.pt')
    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    log.logger.info(f'Save the best model as clintox_model.pt.\n')
    log.logger.info('Best Epoch: {} | test_ROC-AUC: {:.5f}'.format(best_epoch, best_test_auc))


if __name__ == '__main__':
    main()
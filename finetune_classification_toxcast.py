import os
import copy
from time import time, strftime, localtime
import torch
import torch.optim as opt
from sklearn.metrics import roc_auc_score
from utils import parse_args, Logger, set_seed
from torch import nn
from dataloader.data_process_toxcast import MolGraphDataset, molgraph_collate_fn, suppress_output
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
    y_pred_all = []
    y_val_all = []
    y_mask_all = []

    with torch.no_grad():
        batch_idx_val = 0
        for token_idx, atom_mask, target, target_mask, x, edge_index, edge_attr, batch in loader:
            token_idx, atom_mask = token_idx.to(device), atom_mask.to(device)
            target, target_mask = target.to(device), target_mask.to(device)
            x, edge_index, edge_attr, batch = (x.to(device),
                                               edge_index.to(device),
                                               edge_attr.to(device),
                                               batch.to(device))

            batch_idx_val += 1

            out = model(token_idx, x, edge_index, edge_attr, batch)
            out = torch.sigmoid(out)

            if torch.isnan(out).any():
                continue

            y_pred_all.append(out)
            y_val_all.append(target)
            y_mask_all.append(target_mask)

    y_pred_all = torch.cat(y_pred_all)
    y_val_all = torch.cat(y_val_all)
    y_mask_all = torch.cat(y_mask_all)

    aurocs = []
    for i in range(y_val_all.shape[1]):
        label_mask = y_mask_all[:, i].bool()

        if label_mask.sum() > 0:
            try:
                label_pred = y_pred_all[label_mask, i]
                label_true = y_val_all[label_mask, i]

                if torch.unique(label_true).shape[0] > 1:
                    auroc = roc_auc_score(label_true.cpu().numpy(), label_pred.cpu().numpy())
                    aurocs.append(auroc)
            except Exception:
                pass

    if len(aurocs) > 0:
        mean_auroc = np.mean(aurocs)
        return mean_auroc
    else:
        return 0.5


def masked_bce_with_logits_loss(pred, target, mask):
    loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
    masked_loss = loss * mask
    num_valid = mask.sum()
    if num_valid > 0:
        return masked_loss.sum() / num_valid
    else:
        return torch.tensor(0.0, device=pred.device)


def main():
    args = parse_args()
    args.save_path = 'save/'
    set_seed(2025)
    log = Logger(f'{args.save_path}toxcast/', f'toxcast_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')

    args.epochs = 70
    args.gpu = '0'
    args.lr = 7e-5
    args.bs = 16
    args.weight_decay = 1e-5
    args.dropout = 0.2
    args.data = 'toxcast'
    args.verbose = False
    args.num_labels = 15

    early_stop_patience = 15

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with suppress_output():
        full_dataset = MolGraphDataset('/home/ntu/zyk/GRAMMER_BERT/grammer-BERT/data/toxcast/raw/toxcast.csv', num_labels=args.num_labels)

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

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=molgraph_collate_fn,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, collate_fn=molgraph_collate_fn,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, collate_fn=molgraph_collate_fn,
                             num_workers=2, pin_memory=True)

    d_model = 768
    feat_dim = 300
    num_labels = args.num_labels

    one_model = BERT_atom_embedding_generator(d_model=d_model, n_layers=6, vocab_size=83,
                                              maxlen=501, d_k=64, d_v=64, n_heads=12, d_ff=768 * 4,
                                              global_label_dim=1, atom_label_dim=15)
    two_model = GCN(num_layer=5, emb_dim=300, feat_dim=feat_dim, drop_ratio=0.1, pool='mean')

    combined_model = CombinedModel(one_model, two_model)
    predictor_model = Predictor(d_model + feat_dim, num_labels, dropout=args.dropout)

    model = FullModel(combined_model, predictor_model)
    model.to(device)

    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model)

    best_metric = 0
    optimizer = opt.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)

    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.7, patience=7, min_lr=1e-6)

    log.logger.info(f'{"=" * 60} toxcast {"=" * 60}\n'
                    f'Train: {len(train_dataset)}; Val: {len(val_dataset)}; Test: {len(test_dataset)}'
                    f'\nTarget: {args.data}; Batch_size: {args.bs}; Num Labels: {args.num_labels}\nStart Training {"=" * 60}')
    log.logger.info(f'Hyperparameters: LR={args.lr}, Weight Decay={args.weight_decay}, Dropout={args.dropout}')

    t0 = time()
    early_stop = 0
    auc_list = []
    loss_history = []
    lr_history = []

    if not os.path.exists(args.save_path + 'toxcast/'):
        os.makedirs(args.save_path + 'toxcast/')

    scaler = None

    for epoch in range(0, args.epochs):
        model.train()
        loss = 0.0
        t1 = time()
        batch_idx = 0

        for token_idx, atom_mask, target, target_mask, x, edge_index, edge_attr, batch in train_loader:
            token_idx, atom_mask = token_idx.to(device), atom_mask.to(device)
            target, target_mask = target.to(device), target_mask.to(device)
            x, edge_index, edge_attr, batch = (x.to(device),
                                               edge_index.to(device),
                                               edge_attr.to(device),
                                               batch.to(device))

            batch_idx += 1
            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    out = model(token_idx, x, edge_index, edge_attr, batch)
                    if torch.isnan(out).any():
                        continue
                    loss_batch = masked_bce_with_logits_loss(out, target, target_mask)

                scaler.scale(loss_batch).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(token_idx, x, edge_index, edge_attr, batch)
                if torch.isnan(out).any():
                    continue
                loss_batch = masked_bce_with_logits_loss(out, target, target_mask)
                loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            loss += loss_batch.item() / args.bs

        metric = run_eval(args, model, val_loader, device)
        metric_test = run_eval(args, model, test_loader, device)
        auc_list.append(metric_test)

        log.logger.info(
            'Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | val_ROC-AUC: {:.5f} | test_ROC-AUC: {:.5f}'
            '| Lr: {:.3f}'.format(epoch + 1, time() - t1, loss * 1e4, metric, metric_test,
                                  optimizer.param_groups[0]['lr'] * 1e5))
        loss_history.append(loss * 1e4)
        lr_history.append(optimizer.param_groups[0]['lr'])
        lr_scheduler.step(metric)

        if metric > best_metric:
            best_metric = metric
            best_model = copy.deepcopy(model)
            best_epoch = epoch + 1
            early_stop = 0
        else:
            early_stop += 1

        if early_stop >= early_stop_patience:
            log.logger.info(f'Early Stopping!!! No Improvement on Validation Metric for {early_stop_patience} Epochs.')
            break

    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    checkpoint = {'epochs': args.epochs}

    if len(args.gpu) > 1:
        checkpoint['model'] = best_model.module.state_dict()
    else:
        checkpoint['model'] = best_model.state_dict()

    torch.save(checkpoint, args.save_path + f'toxcast/toxcast_model.pt')
    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    log.logger.info(f'Save the best model as toxcast_model.pt.\n')


if __name__ == '__main__':
    main()
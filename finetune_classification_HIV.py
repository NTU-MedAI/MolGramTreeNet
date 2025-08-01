import os
import copy
import numpy as np
from time import time, strftime, localtime
import torch
import torch.optim as opt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from utils import parse_args, Logger, set_seed
from torch import nn
from dataloader.data_process_HIV import MolGraphDataset, molgraph_collate_fn, suppress_output
from torch.utils.data import DataLoader, random_split, Subset
from model.my_nn import BERT_atom_embedding_generator
from model.gcn_finetune import GCN
from model.my_fusion_model import CombinedModel
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class FullModel(nn.Module):
    def __init__(self, combined_model, predictor):
        super().__init__()
        self.combined_model = combined_model
        self.predictor = predictor

    def forward(self, token_idx, x, edge_index, edge_attr, batch):
        combined_out = self.combined_model(token_idx, x, edge_index, edge_attr, batch)
        return self.predictor(combined_out)


class predictor(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.BatchNorm1d(dim * 2),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(dim * 2, 1)
        )

    def forward(self, feats):
        return self.out(feats)


def run_eval(model, loader, device):
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
            out = out.squeeze()
            out = torch.sigmoid(out)

            target = target.squeeze()

            if torch.isnan(out).any():
                continue

            y_pred.append(out)
            y_val.append(target)

    if not y_pred or not y_val:
        return 0.5

    y_pred = torch.cat(y_pred)
    y_val = torch.cat(y_val)

    try:
        auroc = roc_auc_score(y_val.cpu().numpy(), y_pred.cpu().numpy())
    except ValueError:
        auroc = 0.5

    try:
        precision, recall, _ = precision_recall_curve(y_val.cpu().numpy(), y_pred.cpu().numpy())
        auprc = auc(recall, precision)
    except:
        auprc = 0.0

    return auroc, auprc


def plot_training_curves(loss_history, val_auc_history, test_auc_history, val_auprc_history, test_auprc_history,
                         lr_history, save_dir):
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    plt.plot(loss_history, label='Training Loss', color='tab:blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(val_auc_history, label='Validation AUC-ROC', color='tab:green')
    plt.plot(test_auc_history, label='Test AUC-ROC', color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel('AUC-ROC')
    plt.title('AUC-ROC Curves')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(val_auprc_history, label='Validation AUPRC', color='tab:olive')
    plt.plot(test_auprc_history, label='Test AUPRC', color='tab:purple')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.title('AUPRC Curves')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(lr_history, label='Learning Rate', color='tab:orange')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()


def main():
    args = parse_args()
    args.save_path = 'save/'

    set_seed(2000)

    log = Logger(f'{args.save_path}HIV/', f'HIV_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')

    args.epochs = 60
    args.gpu = '0'
    args.lr = 4e-5 * len(args.gpu.split(','))
    args.bs = 64 * len(args.gpu.split(','))
    args.weight_decay = 1e-6
    args.data = 'p_np'
    args.num_workers = 4

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with suppress_output():
        full_dataset = MolGraphDataset('/home/ntu/Documents/zyk/MolGramTreeNet/data/HIV/raw/hiv.csv')

    length = len(full_dataset)
    indices = list(range(length))

    generator = torch.Generator().manual_seed(2000)

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

    train_indices, val_indices, test_indices = random_split(indices,
                                                            lengths=[train, val, test],
                                                            generator=generator)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                              collate_fn=molgraph_collate_fn,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)

    val_loader = DataLoader(val_dataset, batch_size=args.bs,
                            collate_fn=molgraph_collate_fn,
                            num_workers=args.num_workers,
                            pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=args.bs,
                             collate_fn=molgraph_collate_fn,
                             num_workers=args.num_workers,
                             pin_memory=True)

    d_model = 768
    feat_dim = 300

    one_model = BERT_atom_embedding_generator(d_model=d_model, n_layers=6, vocab_size=83,
                                              maxlen=501, d_k=64, d_v=64, n_heads=12, d_ff=768 * 4,
                                              global_label_dim=1, atom_label_dim=15)

    two_model = GCN(num_layer=5, emb_dim=300, feat_dim=feat_dim, drop_ratio=0.15, pool='mean')

    combined_model = CombinedModel(one_model, two_model)
    Predictor = predictor(d_model + feat_dim, dropout=0.2)

    model = FullModel(combined_model, Predictor)
    model.to(device)

    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model)

    best_metric = 0
    criterion = torch.nn.BCELoss()

    optimizer = opt.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                         weight_decay=args.weight_decay)

    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', factor=0.7, patience=5, min_lr=1e-6, verbose=True
    )

    log.logger.info(f'{"=" * 60} HIV {"=" * 60}\n'
                    f'Train: {len(train_dataset)}; Val: {len(val_dataset)}; Test: {len(test_dataset)}'
                    f'\nTarget: {args.data}; Batch_size: {args.bs}; Learning Rate: {args.lr}'
                    f'\nWeight Decay: {args.weight_decay}\nStart Training {"=" * 60}')

    t0 = time()
    early_stop = 0
    early_stop_patience = 20

    loss_history = []
    val_auc_history = []
    val_auprc_history = []
    test_auc_history = []
    test_auprc_history = []
    lr_history = []
    best_epoch = 0
    best_model = None
    best_test_auc = 0

    try:
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
                out = out.squeeze()
                out = torch.sigmoid(out)

                target = target.squeeze()

                if torch.isnan(out).any():
                    continue

                loss_batch = criterion(out, target.float())
                loss += loss_batch.item() / len(train_loader)

                loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            val_auroc, val_auprc = run_eval(model, val_loader, device)
            test_auroc, test_auprc = run_eval(model, test_loader, device)

            loss_history.append(loss * 1e4)
            val_auc_history.append(val_auroc)
            val_auprc_history.append(val_auprc)
            test_auc_history.append(test_auroc)
            test_auprc_history.append(test_auprc)
            lr_history.append(optimizer.param_groups[0]['lr'])

            lr_scheduler.step(val_auroc)

            log.logger.info(
                'Epoch: {:3d} | Time: {:.1f}s | Loss: {:.4f} | val_ROC-AUC: {:.4f} | val_PR-AUC: {:.4f} | '
                'test_ROC-AUC: {:.4f} | test_PR-AUC: {:.4f} | '
                'Lr: {:.6f}'.format(epoch + 1, time() - t1, loss * 1e4, val_auroc, val_auprc,
                                    test_auroc, test_auprc, optimizer.param_groups[0]['lr']))

            if val_auroc > best_metric:
                best_metric = val_auroc
                best_model = copy.deepcopy(model)
                best_epoch = epoch + 1
                best_test_auc = test_auroc
                early_stop = 0
            else:
                early_stop += 1

            if early_stop >= early_stop_patience:
                log.logger.info(
                    f'Early Stopping! No Improvement on Validation AUC-ROC for {early_stop_patience} Epochs.')
                break

            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict() if len(args.gpu) <= 1 else model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_auroc': val_auroc,
                    'test_auroc': test_auroc
                }
                torch.save(checkpoint, f"{args.save_path}HIV/checkpoint_epoch{epoch + 1}.pt")

    except KeyboardInterrupt:
        log.logger.info('Training is interrupted.')
    except Exception as e:
        log.logger.error(f'Error occurred: {str(e)}')

    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))

    checkpoint = {
        'epochs': args.epochs,
        'best_epoch': best_epoch,
        'best_val_auroc': best_metric
    }

    if best_model is not None:
        if len(args.gpu) > 1:
            checkpoint['model'] = best_model.module.state_dict()
        else:
            checkpoint['model'] = best_model.state_dict()
        torch.save(checkpoint, args.save_path + f'HIV_best_model.pt')

    final_checkpoint = {'epochs': args.epochs}
    if len(args.gpu) > 1:
        final_checkpoint['model'] = model.module.state_dict()
    else:
        final_checkpoint['model'] = model.state_dict()
    torch.save(final_checkpoint, args.save_path + f'HIV_final_model.pt')

    log.logger.info(f'Save the best model as HIV_best_model.pt and final model as HIV_final_model.pt')
    log.logger.info('Best Epoch: {} | Best Validation AUC-ROC: {:.5f}'.format(best_epoch, best_metric))

    plot_training_curves(loss_history, val_auc_history, test_auc_history,
                         val_auprc_history, test_auprc_history,
                         lr_history, args.save_path + 'HIV/')


if __name__ == '__main__':
    main()
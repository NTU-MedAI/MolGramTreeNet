import numpy as np
import os
import copy
from time import time, strftime, localtime
import torch
import torch.optim as opt
import torch.nn.functional as F
from utils import parse_args, Logger
from torch import nn
from dataloader.data_process_lipo import MolGraphDataset, molgraph_collate_fn, suppress_output
from torch.utils.data import DataLoader, Subset
from model.my_nn import BERT_atom_embedding_generator
from model.gcn_finetune import GCN
from model.my_fusion_model import CombinedModel
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')

class FullModel(nn.Module):
    def __init__(self, combined_model, predictor):
        super().__init__()
        self.combined_model = combined_model
        self.predictor = predictor

    def forward(self, token_idx, x, edge_index, edge_attr, batch):
        combined_out = self.combined_model(token_idx, x, edge_index, edge_attr, batch)
        return self.predictor(combined_out)

class EnhancedPredictor(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.dropout_input = nn.Dropout(p=dropout / 2)
        self.bn_input = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.bn1 = nn.BatchNorm1d(dim * 2)
        self.dropout1 = nn.Dropout(p=dropout)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout2 = nn.Dropout(p=dropout)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(dim, 1)

    def forward(self, feats):
        x = self.bn_input(feats)
        x = self.dropout_input(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        return self.fc3(x)

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        abs_error = torch.abs(y_pred - y_true)
        quadratic = torch.min(abs_error, torch.tensor(self.delta, device=y_pred.device))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()

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

            loss = F.mse_loss(out, target, reduction='sum').item()
            total_loss += loss
            total_samples += target.numel()
            y_val.append(target.detach().cpu())
            y_pred.append(out.detach().cpu())

    mean_squared_error = total_loss / total_samples
    rmse = np.sqrt(mean_squared_error)
    return rmse

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    args.save_path = 'save/'

    if not os.path.exists(args.save_path + 'lipo/'):
        os.makedirs(args.save_path + 'lipo/')

    set_seed(42)
    log = Logger(f'{args.save_path}lipo/', f'lipo_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')

    args.epochs = 120
    args.gpu = '0'
    args.lr = 1e-5
    args.bs = 32
    args.weight_decay = 2e-4
    args.dropout = 0.2
    args.grad_clip = 1.0
    args.use_huber = False
    args.huber_delta = 1.0
    args.early_stop = 25
    args.gradient_accumulation = 1
    args.use_amp = False
    args.warmup_epochs = 5

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with suppress_output():
        full_dataset = MolGraphDataset('/home/ntu/Documents/zyk/MolGramTreeNet/data/Lipo/raw/lipo.csv')

    length = len(full_dataset)
    targets = []
    for i in range(length):
        _, _, target, _, _, _ = full_dataset[i]
        targets.append(target.item())
    targets = np.array(targets)

    indices = np.array(range(length))
    train_idx, test_idx, train_targets, _ = train_test_split(
        indices, targets, test_size=0.2, random_state=42
    )

    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.125, random_state=42
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=molgraph_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.bs, collate_fn=molgraph_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, collate_fn=molgraph_collate_fn)

    d_model = 768
    feat_dim = 300
    one_model = BERT_atom_embedding_generator(
        d_model=d_model, n_layers=6, vocab_size=83,
        maxlen=501, d_k=64, d_v=64, n_heads=12, d_ff=768 * 4,
        global_label_dim=1, atom_label_dim=15
    )
    two_model = GCN(num_layer=5, emb_dim=300, feat_dim=feat_dim, drop_ratio=0.15, pool='mean')

    combined_model = CombinedModel(one_model, two_model)
    Predictor = EnhancedPredictor(d_model + feat_dim, dropout=args.dropout)

    model = FullModel(combined_model, Predictor)
    model.to(device)

    if len(args.gpu.split(',')) > 1:
        model = torch.nn.DataParallel(model)

    if args.use_huber:
        criterion = HuberLoss(delta=args.huber_delta)
        log.logger.info(f"Using Huber Loss with delta={args.huber_delta}")
    else:
        criterion = torch.nn.MSELoss()
        log.logger.info("Using MSE Loss")

    optimizer = opt.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    lr_scheduler = opt.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log.logger.info(f'{"=" * 60} Lipo {"=" * 60}')
    log.logger.info(f'Train: {len(train_dataset)}; Val: {len(val_dataset)}; Test: {len(test_dataset)}')
    log.logger.info(f'Hyperparameters: LR={args.lr}, BS={args.bs}, WD={args.weight_decay}, '
                    f'Dropout={args.dropout}, Grad Clip={args.grad_clip}')
    log.logger.info(f'Starting training with {args.epochs} epochs')

    t0 = time()
    early_stop = 0
    best_metric = float('inf')
    best_model = None
    best_epoch = 0

    RMSE_list = []
    loss_history = []
    lr_history = []
    rmse_history = []

    scaler = GradScaler() if args.use_amp else None

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
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

            if args.use_amp:
                with autocast():
                    out = model(token_idx, x, edge_index, edge_attr, batch)

                    if torch.isnan(out).any():
                        continue

                    loss_batch = criterion(out, target.float())

                scaler.scale(loss_batch).backward()

                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(token_idx, x, edge_index, edge_attr, batch)

                if torch.isnan(out).any():
                    continue

                loss_batch = criterion(out, target.float())
                loss_batch.backward()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()

            epoch_loss += loss_batch.item()
            lr_scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        val_rmse = run_eval(model, val_loader, device)
        test_rmse = run_eval(model, test_loader, device)
        RMSE_list.append(test_rmse)
        rmse_history.append(val_rmse)

        log.logger.info(
            f'Epoch: {epoch + 1} | Time: {time() - t1:.1f}s | Loss: {avg_loss * 1e4:.2f} | '
            f'val_RMSE: {val_rmse:.4f} | test_RMSE: {test_rmse:.4f} | '
            f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
        )

        loss_history.append(avg_loss * 1e4)
        lr_history.append(optimizer.param_groups[0]['lr'])

        if val_rmse < best_metric:
            best_metric = val_rmse
            best_model = copy.deepcopy(model)
            best_epoch = epoch + 1
            early_stop = 0
        else:
            early_stop += 1

        if early_stop >= args.early_stop:
            log.logger.info(f'Early Stopping! No improvement for {args.early_stop} epochs.')
            break

    log.logger.info(f'{("=" * 20)} End Training (Time: {(time() - t0) / 3600:.2f}h) {("=" * 20)}')
    best_test_RMSE = min(RMSE_list)
    best_epoch_idx = RMSE_list.index(best_test_RMSE) + 1

    checkpoint = {'epochs': args.epochs}

    if len(args.gpu.split(',')) > 1:
        checkpoint['model'] = best_model.module.state_dict()
    else:
        checkpoint['model'] = best_model.state_dict()

    torch.save(checkpoint, args.save_path + 'lipo/lipo_model.pt')
    log.logger.info(f'Saved best model as lipo_model.pt')
    log.logger.info(f'Best Epoch: {best_epoch} | Best test_RMSE: {best_test_RMSE:.5f}')

if __name__ == '__main__':
    main()
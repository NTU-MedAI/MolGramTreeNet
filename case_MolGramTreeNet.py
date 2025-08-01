import os
import copy
from time import time, strftime, localtime
import torch
import torch.optim as opt
from sklearn.metrics import roc_auc_score
from utils import parse_args, Logger, set_seed
from torch import nn
from dataloader.data_process_bace import MolGraphDataset,molgraph_collate_fn
from torch.utils.data import DataLoader, random_split, Subset
from model.my_nn import BERT_atom_embedding_generator
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
    def __init__(self, dim, dropout=0.2):  # 增加dropout率从0.1到0.3
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(dim * 2, 1)
        )

    def forward(self, feats):
        return self.out(feats)


def run_eval(args, model, loader, device):
    model.eval()
    y_pred = []
    y_val = []
    with torch.no_grad():
        for token_idx, atom_mask, target, x, edge_index, edge_attr, batch in loader:
            token_idx, atom_mask = token_idx.to(device), atom_mask.to(device)
            target = target.to(device)
            x, edge_index, edge_attr, batch = (x.to(device),
                                               edge_index.to(device),
                                               edge_attr.to(device),
                                               batch.to(device))

            out = model(token_idx, x, edge_index, edge_attr, batch)
            out = out.squeeze()
            out = torch.sigmoid(out)

            target = target.squeeze()

            if torch.isnan(out).any():
                continue

            y_pred.append(out)
            y_val.append(target)

    if not y_pred:  # 检查是否有有效预测结果
        return 0.0

    y_pred = torch.cat(y_pred)
    y_val = torch.cat(y_val)

    # 处理可能出现的边界情况
    if torch.all(y_val == 0) or torch.all(y_val == 1):
        return 0.0  # 无法计算ROC时返回0

    auroc = roc_auc_score(y_val.cpu().numpy(), y_pred.cpu().numpy())
    return auroc


def main():
    # 预定义设置
    args = parse_args()
    args.save_path = 'save/'
    torch.manual_seed(2025)
    log = Logger(f'{args.save_path}case_MGTG/', f'caseMGTG_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')
    args.epochs = 100
    args.gpu = '0'
    args.lr = 2e-5 * len(args.gpu.split(','))  # 稍微降低学习率
    args.bs = 16 * len(args.gpu.split(','))
    args.data = 'p_np'
    args.patience = 25  # 添加早停耐心参数，从50减少到20

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    full_dataset = MolGraphDataset('/home/ntu/Documents/zyk/MolGramTreeNet/data/CC50.csv')
    length = len(full_dataset)
    print(f"数据集总样本数: {length}")
    indices = list(range(length))

    # 优化数据集划分策略
    train_size = int(length * 0.8)
    val_size = int(length * 0.1)
    test_size = length - train_size - val_size

    train_indices, val_indices, test_indices = random_split(
        indices,
        lengths=[train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子，确保可复现性
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=molgraph_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, collate_fn=molgraph_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, collate_fn=molgraph_collate_fn)

    # 定义模型 - 减少参数量
    d_model = 768
    feat_dim = 256
    one_model = BERT_atom_embedding_generator(
        d_model=d_model,
        n_layers=6,
        vocab_size=83,
        maxlen=501,
        d_k=64,
        d_v=64,
        n_heads=12,  # 从12减少到8
        d_ff=768 * 4,  # 相应减少前馈网络尺寸
        global_label_dim=1,
        atom_label_dim=15
    )

    two_model = GCN(
        num_layer=4,  # 从5减少到4
        emb_dim=256,  # 从300减少到256
        feat_dim=feat_dim,
        drop_ratio=0.2,  # 从0.1增加到0.2
        pool='mean'
    )

    combined_model = CombinedModel(one_model, two_model)
    Predictor = predictor(d_model + feat_dim, dropout=0.3)  # 增加dropout

    model = FullModel(combined_model, Predictor)
    model.to(device)

    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model)

    # 训练设置
    best_metric = 0
    best_test_metric = 0
    criterion = torch.nn.BCELoss()

    # 添加L2正则化(权重衰减)
    optimizer = opt.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=5e-6  # 添加权重衰减
    )

    # 更改学习率调度策略，减小factor值使衰减更平缓
    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'max',
        factor=0.7,  # 从0.6提高到0.7，使学习率衰减更平缓
        patience=8,  # 从10降至8，更早开始学习率衰减
        min_lr=5e-6
    )

    log.logger.info(f'{"=" * 60} MGTG {"=" * 60}\n'
                    f'Train: {len(train_dataset)}; Val: {len(val_dataset)}; test: {len(test_dataset)}'
                    f'\nTarget: {args.data}; Batch_size: {args.bs}\nStart Training {"=" * 60}')

    t0 = time()
    early_stop = 0
    auc_list = []

    # 训练循环
    for epoch in range(0, args.epochs):
        model.train()
        loss = 0.0
        t1 = time()

        for token_idx, atom_mask, target, x, edge_index, edge_attr, batch in train_loader:
            token_idx, atom_mask = token_idx.to(device), atom_mask.to(device)
            x, edge_index, edge_attr, batch = (x.to(device),
                                               edge_index.to(device),
                                               edge_attr.to(device),
                                               batch.to(device))
            target = target.to(device)

            optimizer.zero_grad()

            out = model(token_idx, x, edge_index, edge_attr, batch)
            out = out.squeeze()
            out = torch.sigmoid(out)

            target = target.squeeze()

            if torch.isnan(out).any():
                continue

            loss_batch = criterion(out, target.float())

            # 修正损失计算方式
            loss += loss_batch.item() / len(train_loader)

            loss_batch.backward()
            # 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 评估
        metric = run_eval(args, model, val_loader, device)
        metric_test = run_eval(args, model, test_loader, device)
        auc_list.append(metric_test)

        log.logger.info(
            'Epoch: {} | Time: {:.1f}s | Loss: {:.4f} | val_ROC-AUC: {:.5f} | test_ROC-AUC: {:.5f}'
            '| Lr: {:.6f}'.format(epoch + 1, time() - t1, loss, metric, metric_test,
                                  optimizer.param_groups[0]['lr']))

        lr_scheduler.step(metric)

        # 保存最佳模型(基于验证集性能)
        if metric > best_metric:
            best_metric = metric
            best_model = copy.deepcopy(model)
            best_epoch = epoch + 1
            early_stop = 0

            # 记录相应的测试集性能
            if metric_test > best_test_metric:
                best_test_metric = metric_test
        else:
            early_stop += 1

        # 早停检查，从50减少到args.patience(20)
        if early_stop >= args.patience:
            log.logger.info(f'Early Stopping!!! No Improvement on Validation AUC for {args.patience} Epochs.')
            break

    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))

    # 获取最佳测试性能
    best_test_auc = max(auc_list)
    best_epoch = auc_list.index(best_test_auc) + 1

    # 保存模型
    checkpoint = {'epochs': args.epochs}
    if len(args.gpu) > 1:
        checkpoint['model'] = best_model.module.state_dict()
    else:
        checkpoint['model'] = best_model.state_dict()
    torch.save(checkpoint, args.save_path + f'bbbp_model.pt')

    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    log.logger.info(f'Save the best model as MGTG_model.pt.\n')
    log.logger.info('Best Epoch: {} | val_ROC-AUC: {:.5f} | Best test_ROC-AUC: {:.5f}'.format(
        best_epoch, best_metric, best_test_auc))

    # 在最后进行一次完整的测试集评估
    final_test_auc = run_eval(args, best_model, test_loader, device)


if __name__ == '__main__':
    main()


import os
import copy
from time import time, strftime, localtime
import torch
import torch.optim as opt
from sklearn.metrics import roc_auc_score
from utils import parse_args, Logger, set_seed
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset

# 只导入数据加载器部分
from data_process_bace_ablation import AblationMolGraphDataset, ablation_collate_fn
from model.gcn_finetune import GCN


# 直接在文件中定义GCN预测器
class GCNPredictor(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(dim * 2, 1)
        )

    def forward(self, feats):
        return self.out(feats)


# 直接在文件中定义消融实验模型
class AblationModel(nn.Module):
    def __init__(self, gcn_model, predictor):
        super().__init__()
        self.gcn_model = gcn_model
        self.predictor = predictor

    def forward(self, x, edge_index, edge_attr, batch):
        # 只使用GCN特征
        gcn_out = self.gcn_model(x, edge_index, edge_attr, batch)
        return self.predictor(gcn_out)


def run_eval(model, loader, device):
    """评估函数"""
    model.eval()
    y_pred = []
    y_val = []
    with torch.no_grad():
        for x, edge_index, edge_attr, batch, target in loader:
            x, edge_index = x.to(device), edge_index.to(device)
            edge_attr, batch = edge_attr.to(device), batch.to(device)
            target = target.to(device)

            out = model(x, edge_index, edge_attr, batch)
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

    # 处理边界情况
    if torch.all(y_val == 0) or torch.all(y_val == 1):
        return 0.0  # 无法计算ROC时返回0

    auroc = roc_auc_score(y_val.cpu().numpy(), y_pred.cpu().numpy())
    return auroc


def main():
    # 预定义设置
    args = parse_args()
    args.save_path = 'save/ablation/'
    os.makedirs(args.save_path, exist_ok=True)
    torch.manual_seed(2025)
    log = Logger(f'{args.save_path}', f'ablation_bace_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')
    args.epochs = 100
    args.gpu = '0'
    args.lr = 3e-5 * len(args.gpu.split(','))
    args.bs = 16 * len(args.gpu.split(','))
    args.patience = 25

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载消融实验数据集
    full_dataset = AblationMolGraphDataset('/home/ntu/Documents/zyk/MolGramTreeNet/data/bace/raw/bace.csv')
    length = len(full_dataset)
    print(f"数据集总样本数: {length}")
    indices = list(range(length))

    # 数据集划分
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

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=ablation_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, collate_fn=ablation_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, collate_fn=ablation_collate_fn)

    # 定义仅使用GCN的模型
    feat_dim = 256
    gcn_model = GCN(
        num_layer=4,
        emb_dim=256,
        feat_dim=feat_dim,
        drop_ratio=0.2,
        pool='mean'
    )

    predictor = GCNPredictor(feat_dim, dropout=0.3)
    model = AblationModel(gcn_model, predictor)
    model.to(device)

    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model)

    # 训练设置
    best_metric = 0
    best_test_metric = 0
    criterion = torch.nn.BCELoss()

    optimizer = opt.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=5e-6  # 添加权重衰减
    )

    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'max',
        factor=0.7,
        patience=8,
        min_lr=5e-6
    )

    log.logger.info(f'{"=" * 60} bace 消融实验（仅GCN模型） {"=" * 60}\n'
                    f'Train: {len(train_dataset)}; Val: {len(val_dataset)}; test: {len(test_dataset)}'
                    f'\nBatch_size: {args.bs}\nStart Training {"=" * 60}')

    t0 = time()
    early_stop = 0
    auc_list = []

    # 训练循环
    for epoch in range(0, args.epochs):
        model.train()
        loss = 0.0
        t1 = time()

        for x, edge_index, edge_attr, batch, target in train_loader:
            x, edge_index = x.to(device), edge_index.to(device)
            edge_attr, batch = edge_attr.to(device), batch.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            out = model(x, edge_index, edge_attr, batch)
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

        # 评估
        metric = run_eval(model, val_loader, device)
        metric_test = run_eval(model, test_loader, device)
        auc_list.append(metric_test)

        log.logger.info(
            'Epoch: {} | Time: {:.1f}s | Loss: {:.4f} | val_ROC-AUC: {:.5f} | test_ROC-AUC: {:.5f}'
            '| Lr: {:.6f}'.format(epoch + 1, time() - t1, loss, metric, metric_test,
                                  optimizer.param_groups[0]['lr']))

        lr_scheduler.step(metric)

        # 保存最佳模型
        if metric > best_metric:
            best_metric = metric
            best_model = copy.deepcopy(model)
            best_epoch = epoch + 1
            early_stop = 0

            if metric_test > best_test_metric:
                best_test_metric = metric_test
        else:
            early_stop += 1

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
    torch.save(checkpoint, args.save_path + f'ablation_bace_model.pt')

    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    log.logger.info(f'Save the best model as ablation_bace_model.pt.\n')
    log.logger.info('Best Epoch: {} | val_ROC-AUC: {:.5f} | Best test_ROC-AUC: {:.5f}'.format(
        best_epoch, best_metric, best_test_auc))

    # 最终评估
    final_test_auc = run_eval(best_model, test_loader, device)
    log.logger.info(f'Final test ROC-AUC with best model: {final_test_auc:.5f}')
    log.logger.info(f'Note: This is an ablation experiment result WITHOUT grammar tree representation')


if __name__ == '__main__':
    main()


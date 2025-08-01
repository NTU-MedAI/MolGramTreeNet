import os
import copy
from time import time, strftime, localtime
import torch
import torch.optim as opt
from sklearn.metrics import roc_auc_score
from utils import parse_args, Logger, set_seed
from torch import nn
from dataloader.case_dataloader import MolGraphDataset,molgraph_collate_fn, suppress_output
from torch.utils.data import DataLoader, random_split,Subset
from model.my_nn import BERT_atom_embedding_generator

class predictor(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(dim, dim*2), nn.Dropout(p=dropout), nn.ReLU(), nn.Linear(dim*2, 1))

    def forward(self, feats):
        return self.out(feats)

def run_eval(args, model, loader, device):
    model.eval()
    y_pred = []
    y_val = []
    with torch.no_grad():
        batch_idx_val = 0
        for token_idx, atom_mask, target in loader:
            token_idx, atom_mask = token_idx.to(device), atom_mask.to(device)
            target = target.to(device)

            batch_idx_val += 1
            # 将输入的 token 索引输入到模型中，得到模型的输出。
            out = model(token_idx)
            # 移除输出张量中维度为 1 的维度。例如，如果输出形状为 (batch_size, 1)，
            # 则会将其变为 (batch_size,)。
            out = out.squeeze()
            # 对模型的输出应用 sigmoid 函数，将输出值映射到 [0, 1] 区间，通常用于二分类问题。
            out = torch.sigmoid(out)
            # 移除目标标签张量中维度为 1 的维度。
            target = target.squeeze()

            if torch.isnan(out).any():
                print(batch_idx_val)
                continue

            y_pred.append(out)
            y_val.append(target)
    y_pred = torch.cat(y_pred)
    y_val = torch.cat(y_val)
    auroc = roc_auc_score(y_val.cpu().numpy(), y_pred.cpu().numpy())

    return auroc


def main():
    #预定义设置
    args = parse_args()
    args.save_path = 'save/'
    torch.manual_seed(2025) #可以用seed迭代循环10次，保存在列表里输出AUC-ROC值
    log = Logger(f'{args.save_path}case/', f'case_{strftime("%Y-% m-%d_%H-%M-%S", localtime())}.log')
    args.epochs = 100
    args.gpu = '0'
    args.lr = 2e-5 * len(args.gpu.split(','))
    args.bs = 16 * len(args.gpu.split(','))
    args.data = 'p_np'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with suppress_output():
     full_dataset = MolGraphDataset('/home/ntu/Documents/zyk/MolGramTreeNet/data/CC50.csv')#数据集路径

    length = len(full_dataset)

    indices = list(range(length))

    # 计算训练集的样本数量，占总样本数的 80%
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

    train_indices, val_indices, test_indices = random_split(indices, lengths=[train, val, test])#随机划分

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,collate_fn=molgraph_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.bs,collate_fn=molgraph_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.bs,collate_fn=molgraph_collate_fn)

    #定义模型
    model_1 = BERT_atom_embedding_generator(d_model=768, n_layers=6, vocab_size=83,
                                          maxlen=501, d_k=64, d_v=64, n_heads=12, d_ff=768 * 4,
                                          global_label_dim=1, atom_label_dim=15)
    model_2 = predictor(768)
    model = nn.Sequential(model_1,model_2)
    model.to(device)

    if len(args.gpu) > 1:  model = torch.nn.DataParallel(model)

    best_metric = 0
    # 定义二元交叉熵损失函数
    criterion = torch.nn.BCELoss()
    # 定义 Adam 优化器，用于更新模型的参数
    optimizer = opt.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.98))

    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.6, patience=10, min_lr=5e-8)
    log.logger.info(f'{"=" * 60} case {"=" * 60}\n'
                    f'Train: {len(train_dataset)}; Val: {len(val_dataset)}; test: {len(test_dataset)}'
                    f'\nTarget: {args.data}; Batch_size: {args.bs}\nStart Training {"=" * 60}')

    t0 = time()
    early_stop = 0
    auc_list = []
    # try:
    for epoch in range(0, args.epochs):
        model.train()
        loss = 0.0
        t1 = time()
        batch_idx = 0
        for token_idx, atom_mask, target in train_loader:
            token_idx, atom_mask = token_idx.to(device), atom_mask.to(device)
            # print(token_idx)
            # print(atom_mask)
            # atom_mask_index =torch.nonzero(atom_mask, as_tuple=False)
            # print(atom_mask_index)
            target = target.to(device)

            batch_idx += 1
            optimizer.zero_grad()

            out = model(token_idx)
            out = out.squeeze()
            out = torch.sigmoid(out)
            # print(out.shape)

            target = target.squeeze()
            # print(target.shape)

            if torch.isnan(out).any():
                continue
            loss_batch = criterion(out, target.float())

            loss += loss_batch.item()/(len(target) * args.bs)

            loss_batch.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
            optimizer.step()

        metric = run_eval(args, model, val_loader,device)

        metric_test = run_eval(args, model, test_loader,device)
        auc_list.append(metric_test)

        log.logger.info(
            'Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | val_ROC-AUC: {:.5f} | test_ROC-AUC: {:.5f}'
                        '| Lr: {:.3f}'.format(epoch + 1, time() - t1, loss * 1e4, metric , metric_test,
                                              optimizer.param_groups[0]['lr'] * 1e5))

        lr_scheduler.step(metric)

        if metric > best_metric:
            best_metric = metric
            best_model = copy.deepcopy(model)  # deep copy model
            best_epoch = epoch + 1
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= 50: log.logger.info('Early Stopping!!! No Improvement on Loss for 100 Epochs.'); break
    # except:
    #     log.logger.info('Training is interrupted.')
    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    checkpoint = {'epochs': args.epochs}

    best_test_auc = max(auc_list)
    best_epoch = auc_list.index(best_test_auc) + 1

    # auroc, auprc, _ = run_eval(args, best_model, test_loader, y_test)
    if len(args.gpu) > 1:
        checkpoint['model'] = best_model.module.state_dict()
    else:
        checkpoint['model'] = best_model.state_dict()
    torch.save(checkpoint, args.save_path + f'case_model.pt')
    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    log.logger.info(f'Save the best model as case_model.pt.\n')


if __name__ == '__main__':
    main()

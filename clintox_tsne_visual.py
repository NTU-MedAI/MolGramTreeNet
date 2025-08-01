import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib as mpl
import random

# 设置更优雅的绘图风格，但禁用网格线
plt.style.use('default')  # 使用默认风格避免网格线
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

# Import your model classes and data loaders
from dataloader.data_process_clintox import MolGraphDataset, molgraph_collate_fn
from model.my_nn import BERT_atom_embedding_generator
from model.gcn_finetune import GCN
from model.my_fusion_model import CombinedModel
from utils import parse_args


# FullModel class for ClinTox
class FullModel(torch.nn.Module):
    def __init__(self, combined_model, predictor):
        super().__init__()
        self.combined_model = combined_model
        self.predictor = predictor

    def forward(self, token_idx, x, edge_index, edge_attr, batch):
        combined_out = self.combined_model(token_idx, x, edge_index, edge_attr, batch)
        return self.predictor(combined_out)

    def get_embeddings(self, token_idx, x, edge_index, edge_attr, batch):
        # This method returns the embeddings before the prediction layer
        return self.combined_model(token_idx, x, edge_index, edge_attr, batch)


# Predictor class for ClinTox (with 2 output labels)
class Predictor(torch.nn.Module):
    def __init__(self, dim, num_labels=2, dropout=0.1):
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2),
            torch.nn.Dropout(p=dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(dim * 2, num_labels)
        )

    def forward(self, feats):
        return self.out(feats)


def initialize_model(device):
    """Initialize a new model with random weights"""
    d_model = 768
    feat_dim = 300  # Changed to match the training script
    num_labels = 2  # Two labels for ClinTox (FDA_APPROVED and CT_TOX)

    one_model = BERT_atom_embedding_generator(
        d_model=d_model,
        n_layers=6,
        vocab_size=83,
        maxlen=501,
        d_k=64,
        d_v=64,
        n_heads=12,
        d_ff=768 * 4,
        global_label_dim=1,
        atom_label_dim=15
    )

    two_model = GCN(
        num_layer=5,
        emb_dim=300,
        feat_dim=feat_dim,
        drop_ratio=0.1,  # Changed to match the training script
        pool='mean'
    )

    combined_model = CombinedModel(one_model, two_model)
    predictor_model = Predictor(d_model + feat_dim, num_labels)

    model = FullModel(combined_model, predictor_model)
    model.to(device)

    return model


def load_trained_model(device, model_path):
    """Load a trained model from checkpoint"""
    # First initialize the model architecture
    model = initialize_model(device)

    # Load the saved state dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    return model


def fix_datatypes_for_gcn(x, edge_index, edge_attr, batch):
    """Helper function to fix data types before passing to the model"""
    # Fix node features - must be Long for embedding layers
    if x.dtype != torch.long:
        x = x.long()

    # Fix edge attributes - must be Long for embedding layers
    if edge_attr.dtype != torch.long:
        edge_attr = edge_attr.long()

    # Edge index should also be Long
    if edge_index.dtype != torch.long:
        edge_index = edge_index.long()

    # Batch indices should be Long
    if batch.dtype != torch.long:
        batch = batch.long()

    return x, edge_index, edge_attr, batch


def extract_embeddings(model, data_loader, device, label_index=0):
    """Extract embeddings and specific label from the model"""
    model.eval()
    embeddings = []
    labels = []
    indices = []  # 添加索引跟踪

    batch_start_idx = 0

    with torch.no_grad():
        for token_idx, atom_mask, target, x, edge_index, edge_attr, batch in tqdm(data_loader,
                                                                                  desc="Extracting embeddings"):
            # Move tensors to device
            token_idx = token_idx.to(device)
            atom_mask = atom_mask.to(device)
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            batch = batch.to(device)

            batch_size = token_idx.size(0)
            # 记录这个批次中每个分子的索引
            batch_indices = list(range(batch_start_idx, batch_start_idx + batch_size))
            batch_start_idx += batch_size

            # Fix data types for GCN
            x, edge_index, edge_attr, batch = fix_datatypes_for_gcn(x, edge_index, edge_attr, batch)

            try:
                # Get embeddings
                batch_embeddings = model.get_embeddings(token_idx, x, edge_index, edge_attr, batch)

                # Store embeddings and labels
                embeddings.append(batch_embeddings.cpu().numpy())
                labels.append(target[:, label_index].cpu().numpy())
                indices.extend(batch_indices)  # 添加到索引列表
            except Exception as e:
                print(f"Error processing batch: {e}")
                print(f"x shape: {x.shape}, dtype: {x.dtype}")
                print(f"edge_attr shape: {edge_attr.shape}, dtype: {edge_attr.dtype}")
                print(f"edge_index shape: {edge_index.shape}, dtype: {edge_index.dtype}")
                print(f"batch shape: {batch.shape}, dtype: {batch.dtype}")
                raise e

    # Concatenate all batches
    if embeddings:
        embeddings = np.vstack(embeddings)
        labels = np.concatenate(labels)
        indices = np.array(indices)
        return embeddings, labels, indices
    else:
        raise RuntimeError("No valid embeddings were extracted. Check the data processing.")


def plot_tsne(embeddings, labels, indices, title, ax, label_name, tracked_point_idx=None):
    """Plot t-SNE visualization with enhanced aesthetics and class statistics"""
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # 记录被跟踪点的原始和归一化坐标
    tracked_point_info = None
    if tracked_point_idx is not None:
        # 找到跟踪点在当前索引数组中的位置
        point_pos = np.where(indices == tracked_point_idx)[0]
        if len(point_pos) > 0:
            pos = point_pos[0]
            raw_coords = reduced_embeddings[pos]
            tracked_point_info = {
                'index': tracked_point_idx,
                'raw_coords': raw_coords.copy(),  # 保存未归一化的坐标
            }

    # 计算类别统计信息
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_count = np.sum(pos_mask)
    neg_count = np.sum(neg_mask)
    total_count = len(labels)
    pos_percent = (pos_count / total_count) * 100
    neg_percent = (neg_count / total_count) * 100

    # 确定哪个是少数类
    is_pos_minority = pos_count < neg_count

    # Customize labels based on the task
    if label_name == "FDA_APPROVED":
        pos_label = "FDA Approved"
        neg_label = "Not Approved"
        pos_color = '#3366CC'  # 蓝色
        neg_color = '#CC3366'  # 红色
    else:  # CT_TOX
        pos_label = "Non-Toxic"
        neg_label = "Toxic"
        pos_color = '#33AA55'  # 绿色
        neg_color = '#EE3333'  # 红色

    # 根据少数类情况调整点大小
    pos_size = 150 if is_pos_minority else 100
    neg_size = 100 if is_pos_minority else 150

    # 清除当前轴的所有内容
    ax.clear()

    # 禁用网格
    ax.grid(False)

    # 归一化函数 - 将t-SNE结果缩放到0-100范围
    def normalize_to_range(values, min_val, max_val, new_min=0, new_max=100):
        return ((values - min_val) / (max_val - min_val) * (new_max - new_min)) + new_min

    # 计算min/max并归一化
    x_min, x_max = reduced_embeddings[:, 0].min(), reduced_embeddings[:, 0].max()
    y_min, y_max = reduced_embeddings[:, 1].min(), reduced_embeddings[:, 1].max()
    normalized_x = normalize_to_range(reduced_embeddings[:, 0], x_min, x_max)
    normalized_y = normalize_to_range(reduced_embeddings[:, 1], y_min, y_max)

    # 更新跟踪点的归一化坐标
    if tracked_point_info is not None:
        pos = np.where(indices == tracked_point_idx)[0][0]
        tracked_point_info['normalized_coords'] = [normalized_x[pos], normalized_y[pos]]

    # 绘制散点图（使用归一化后的数据）
    if is_pos_minority:
        ax.scatter(
            normalized_x[neg_mask],
            normalized_y[neg_mask],
            s=neg_size, color=neg_color, alpha=0.6,
            label=f"{neg_label} (n={neg_count}, {neg_percent:.1f}%)",
            edgecolors='white', linewidths=0.5
        )
        ax.scatter(
            normalized_x[pos_mask],
            normalized_y[pos_mask],
            s=pos_size, color=pos_color, alpha=0.8,
            label=f"{pos_label} (n={pos_count}, {pos_percent:.1f}%)",
            edgecolors='white', linewidths=0.8
        )
    else:
        ax.scatter(
            normalized_x[pos_mask],
            normalized_y[pos_mask],
            s=pos_size, color=pos_color, alpha=0.6,
            label=f"{pos_label} (n={pos_count}, {pos_percent:.1f}%)",
            edgecolors='white', linewidths=0.5
        )
        ax.scatter(
            normalized_x[neg_mask],
            normalized_y[neg_mask],
            s=neg_size, color=neg_color, alpha=0.8,
            label=f"{neg_label} (n={neg_count}, {neg_percent:.1f}%)",
            edgecolors='white', linewidths=0.8
        )

    # 设置刻度值
    ticks = [0, 20, 40, 60, 80, 100]

    # 特别处理坐标轴的标签，只在左下角显示一个0
    # 先设置刻度位置
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # 然后设置标签文本
    x_labels = ['0', '20', '40', '60', '80', '100']
    y_labels = ['', '20', '40', '60', '80', '100']  # y轴的0标签设为空字符串

    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_yticklabels(y_labels, fontsize=12)

    # 移除刻度线
    ax.tick_params(axis='both', which='both', length=0)

    # 设置坐标轴标签
    ax.set_xlabel("t-SNE Dimension 1", fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=14, fontweight='bold', labelpad=10)

    # 仅显示左侧和底部的坐标轴
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # 加粗坐标轴线
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # 设置坐标轴范围，确保包含0-100
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

    # 标题
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)

    # 图例
    legend = ax.legend(fontsize=16, markerscale=1.5, loc='upper right',
                       frameon=True, facecolor='white', framealpha=0.9, edgecolor='gray')

    # 加粗图例文字
    for text in legend.get_texts():
        text.set_fontweight('bold')

    return tracked_point_info


def main():
    # Parse arguments
    args = parse_args()
    args.bs = 16  # Match batch size from training
    args.gpu = '0'

    # 设置当前日期时间和用户名
    current_datetime = "2025-07-21 01:27:05"
    current_user = "zyk20"

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading ClinTox dataset...")
    full_dataset = MolGraphDataset('/home/ntu/Documents/zyk/MolGramTreeNet/data/ClinTox/raw/clintox.csv')
    print(f"Dataset loaded with {len(full_dataset)} molecules")

    # Use test set for visualization
    test_size = len(full_dataset) // 5  # 20% for test
    indices = list(range(len(full_dataset)))

    # Use the same random seed as in training script
    torch.manual_seed(42)
    np.random.seed(42)
    np.random.shuffle(indices)
    test_indices = indices[:test_size]

    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        collate_fn=molgraph_collate_fn
    )

    # ===== Initialize models =====
    # Untrained model
    print("Initializing untrained model...")
    untrained_model = initialize_model(device)

    # Trained model
    print("\nLoading trained model...")
    model_path = '/home/ntu/Documents/zyk/MolGramTreeNet/save/clintox/clintox_model.pt'  # Path updated to match training script
    try:
        trained_model = load_trained_model(device, model_path)
        print("Trained model loaded successfully")
    except Exception as e:
        print(f"Error loading trained model: {e}")
        print("Please make sure the model file exists and has the correct format")
        return

    # ===== Process each label separately =====
    labels = ["FDA_APPROVED", "CT_TOX"]

    # 创建图形并禁用默认网格
    plt.rcParams['axes.grid'] = False
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))

    # 用于存储要跟踪的点
    tracked_points = {}

    # 处理第一个任务 (FDA_APPROVED)
    print(f"\nProcessing {labels[0]} task...")

    # 为第一个任务提取嵌入
    print(f"Extracting embeddings from untrained model for {labels[0]}...")
    untrained_embeddings, untrained_labels, untrained_indices = extract_embeddings(
        untrained_model, test_loader, device, label_index=0)

    # 在FDA_APPROVED中随机选择一个正样本点（FDA批准的）进行跟踪
    positive_indices = untrained_indices[untrained_labels == 1]
    if len(positive_indices) > 0:
        # 设置随机种子以便结果可重现
        random.seed(42)
        fda_tracked_idx = random.choice(positive_indices)
    else:
        fda_tracked_idx = random.choice(untrained_indices)

    print(f"Selected point with index {fda_tracked_idx} for tracking in {labels[0]}")

    # 绘制未训练模型的图并记录跟踪点信息
    fda_before_info = plot_tsne(
        untrained_embeddings, untrained_labels, untrained_indices,
        f"Before Training - {labels[0]}", axes[0, 0], labels[0],
        tracked_point_idx=fda_tracked_idx)

    print(f"Extracting embeddings from trained model for {labels[0]}...")
    trained_embeddings, trained_labels, trained_indices = extract_embeddings(
        trained_model, test_loader, device, label_index=0)

    # 绘制训练后模型的图并记录跟踪点信息
    fda_after_info = plot_tsne(
        trained_embeddings, trained_labels, trained_indices,
        f"After Training - {labels[0]}", axes[0, 1], labels[0],
        tracked_point_idx=fda_tracked_idx)

    # 处理第二个任务 (CT_TOX)
    print(f"\nProcessing {labels[1]} task...")

    # 为第二个任务提取嵌入
    print(f"Extracting embeddings from untrained model for {labels[1]}...")
    untrained_embeddings, untrained_labels, untrained_indices = extract_embeddings(
        untrained_model, test_loader, device, label_index=1)

    # 在CT_TOX中随机选择一个正样本点（非毒性的）进行跟踪
    positive_indices = untrained_indices[untrained_labels == 1]
    if len(positive_indices) > 0:
        # 使用不同种子选择不同点
        random.seed(43)
        ct_tracked_idx = random.choice(positive_indices)
    else:
        ct_tracked_idx = random.choice(untrained_indices)

    print(f"Selected point with index {ct_tracked_idx} for tracking in {labels[1]}")

    # 绘制未训练模型的图并记录跟踪点信息
    ct_before_info = plot_tsne(
        untrained_embeddings, untrained_labels, untrained_indices,
        f"Before Training - {labels[1]}", axes[1, 0], labels[1],
        tracked_point_idx=ct_tracked_idx)

    print(f"Extracting embeddings from trained model for {labels[1]}...")
    trained_embeddings, trained_labels, trained_indices = extract_embeddings(
        trained_model, test_loader, device, label_index=1)

    # 绘制训练后模型的图并记录跟踪点信息
    ct_after_info = plot_tsne(
        trained_embeddings, trained_labels, trained_indices,
        f"After Training - {labels[1]}", axes[1, 1], labels[1],
        tracked_point_idx=ct_tracked_idx)

    # 输出跟踪点在训练前后的坐标变化
    print("\n" + "=" * 50)
    print(f"Tracked Point Coordinates (FDA_APPROVED, index {fda_tracked_idx}):")
    if fda_before_info and fda_after_info:
        print(
            f"  Before Training - Raw: [{fda_before_info['raw_coords'][0]:.4f}, {fda_before_info['raw_coords'][1]:.4f}]")
        print(
            f"                  - Normalized: [{fda_before_info['normalized_coords'][0]:.2f}, {fda_before_info['normalized_coords'][1]:.2f}]")
        print(
            f"  After Training  - Raw: [{fda_after_info['raw_coords'][0]:.4f}, {fda_after_info['raw_coords'][1]:.4f}]")
        print(
            f"                  - Normalized: [{fda_after_info['normalized_coords'][0]:.2f}, {fda_after_info['normalized_coords'][1]:.2f}]")

    print("\n" + "=" * 50)
    print(f"Tracked Point Coordinates (CT_TOX, index {ct_tracked_idx}):")
    if ct_before_info and ct_after_info:
        print(
            f"  Before Training - Raw: [{ct_before_info['raw_coords'][0]:.4f}, {ct_before_info['raw_coords'][1]:.4f}]")
        print(
            f"                  - Normalized: [{ct_before_info['normalized_coords'][0]:.2f}, {ct_before_info['normalized_coords'][1]:.2f}]")
        print(f"  After Training  - Raw: [{ct_after_info['raw_coords'][0]:.4f}, {ct_after_info['raw_coords'][1]:.4f}]")
        print(
            f"                  - Normalized: [{ct_after_info['normalized_coords'][0]:.2f}, {ct_after_info['normalized_coords'][1]:.2f}]")
    print("=" * 50)

    # 调整：将主标题位置提高，避免与子图标题重叠
    plt.suptitle("t-SNE Visualization of ClinTox Molecule Embeddings",
                 fontsize=28, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.06, hspace=0.35, wspace=0.15)

    # 保存高分辨率图像
    save_path = "clintox_tsne_comparison.png"
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"Combined visualization saved as '{save_path}'")


if __name__ == "__main__":
    main()
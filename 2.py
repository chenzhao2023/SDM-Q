import pandas as pd
import numpy as np
import torch
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import os
script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory

# =================== Parameters ===================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_path = os.path.join(script_dir, "DataSets", "DataSets", "ROSMAP")
N_STAGE = 3
GAMMA = 1.0

# Switch True / False to choose evaluation data
USE_TRAIN = False  # True: Evaluate training set, False: Evaluate test set


# ---------- Data loading ----------
def load_feat(feat_csv, data_csv):
    """Load features only, ensure column count is consistent with training set"""
    feat = pd.read_csv(feat_csv, header=None).squeeze().tolist()
    data = pd.read_csv(data_csv, header=None).dropna(axis=1)
    if len(feat) < data.shape[1]:
        data = data.iloc[:, :len(feat)]
    data.columns = feat[:data.shape[1]]
    return data.values.astype(np.float32)


if USE_TRAIN:
    X1 = load_feat(f"{base_path}/1_featname.csv", f"{base_path}/1_tr.csv")
    X2 = load_feat(f"{base_path}/2_featname.csv", f"{base_path}/2_tr.csv")
    X3 = load_feat(f"{base_path}/3_featname.csv", f"{base_path}/3_tr.csv")
    labels = pd.read_csv(f"{base_path}/labels_tr.csv", header=None).squeeze().astype(int).values
else:
    X1 = load_feat(f"{base_path}/1_featname.csv", f"{base_path}/1_te.csv")
    X2 = load_feat(f"{base_path}/2_featname.csv", f"{base_path}/2_te.csv")
    X3 = load_feat(f"{base_path}/3_featname.csv", f"{base_path}/3_te.csv")
    try:
        labels = pd.read_csv(f"{base_path}/labels_te.csv", header=None).squeeze().astype(int).values
    except FileNotFoundError:
        print("Test labels not found, will skip accuracy and F1 evaluation")
        labels = None

n_samples = X1.shape[0]
stage_dims = [X1.shape[1], X2.shape[1], X3.shape[1]]


# ---------- Network structure ----------
class StageNet(torch.nn.Module):
    def __init__(self, stage_dims, n_class):
        super().__init__()
        self.encoders = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU()
            ) for d in stage_dims
        ])
        fusion_dim = 32 * len(stage_dims)
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(fusion_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU()
        )
        self.q_head = torch.nn.Linear(64, 2)
        self.clf_head = torch.nn.Linear(64, n_class)

    def forward(self, xs):
        embs = [enc(x) for enc, x in zip(self.encoders, xs)]
        x = torch.cat(embs, dim=1)
        z = self.backbone(x)
        q = self.q_head(z)
        log = self.clf_head(z)
        return q, log


# ---------- Build state ----------
def build_state(stage, i):
    states = []
    for idx, X in enumerate([X1, X2, X3]):
        if idx <= stage:
            states.append(torch.from_numpy(X[i]).unsqueeze(0).to(device))
        else:
            states.append(torch.zeros((1, X.shape[1]), dtype=torch.float32, device=device))
    return states


def build_state_by_mask(mask, i):
    """
    mask: A list/array of length 3, 1 indicates the modality is unlocked, 0 indicates unlocked
    returns: [x1_tensor, x2_tensor, x3_tensor]; unlocked ones are filled with zeros
    """
    states = []
    for used, X in zip(mask, [X1, X2, X3]):
        if used == 1:
            states.append(torch.from_numpy(X[i]).unsqueeze(0).to(device))
        else:
            states.append(torch.zeros((1, X.shape[1]), dtype=torch.float32, device=device))
    return states


# ---------- Instantiate and load ----------
n_class = int(np.max(labels)) + 1 if labels is not None else 3
c2_candidates = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
c3_candidates = [0.2, 0.3, 0.4, 0.5, 0.6]

results = []

for c2 in c2_candidates:
    for c3 in c3_candidates:
        print(f"\n>>> Testing COST_MAP = {{1:0.0, 2:{c2}, 3:{c3}}}")

        # Reload networks
        nets = [StageNet(stage_dims, n_class).to(device) for _ in range(N_STAGE)]
        for s in range(N_STAGE):
            path = os.path.join(script_dir, f"StageNet_c2{c2}_c3{c3}_stage{s + 1}.pth")
            if not os.path.exists(path):
                print(f"⚠️ Model file not found: {path}")
                continue
            nets[s].load_state_dict(torch.load(path, map_location=device))
            nets[s].eval()

        # ========= Run inference =========
        used_stage, pred_all, pred_probs = [], [], []
        with torch.no_grad():
            for i in range(n_samples):
                mask = [1, 0, 0]
                while True:
                    curr_k = sum(mask) - 1
                    st = build_state_by_mask(mask, i)
                    q, logits = nets[curr_k](st)
                    action = q.squeeze().argmax().item()

                    if action == 1 or curr_k == N_STAGE - 1:
                        used_stage.append(sum(mask))
                        pred_label = logits.argmax().item()
                        pred_all.append(pred_label)
                        prob = torch.softmax(logits, dim=1).cpu().numpy().flatten()
                        pred_probs.append(prob)
                        break

                    # Continue：next modality selection
                    best_next_mod, best_next_q = None, None
                    for m in range(N_STAGE):
                        if mask[m] == 0:
                            cand_mask = mask.copy()
                            cand_mask[m] = 1
                            cand_st = build_state_by_mask(cand_mask, i)
                            next_k = sum(cand_mask) - 1
                            next_q, _ = nets[next_k](cand_st)
                            cand_value = next_q.max(dim=1)[0].item()
                            if (best_next_q is None) or (cand_value > best_next_q):
                                best_next_q = cand_value
                                best_next_mod = m
                    mask[best_next_mod] = 1

        # ========= Calculate metrics =========
        pred_all = np.array(pred_all)
        labels_np = np.array(labels)

        acc = (pred_all == labels_np).mean()
        f1 = f1_score(labels_np, pred_all, average='weighted', zero_division=0)
        avg_stage = np.mean(used_stage)
        results.append([c2, c3, avg_stage])
        # Statistics of samples at each stage
        stat = Counter(used_stage)
        stage_counts = [stat.get(s, 0) for s in range(1, N_STAGE + 1)]

        # ========= KEY: Stack pred_probs into matrix =========
        pred_probs = np.vstack(pred_probs)  # shape = (n_samples, n_class)

        # ========= KEY: Calculate AUC (universal version, completely correct) =========
        # ========= KEY: Calculate metrics for different classifications =========
        if n_class == 2:
            # Binary classification: Calculate AUC
            auc = roc_auc_score(labels_np, pred_probs[:, 1])
            print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Avg modalities used: {avg_stage:.2f}")
        else:
            # Multi-class: Calculate Macro-F1 (aligns with many multi-class baselines)
            macro_f1 = f1_score(labels_np, pred_all, average='macro', zero_division=0)
            print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Macro-F1: {macro_f1:.4f} | Avg modalities used: {avg_stage:.2f}")

    print(f"Samples stopped at 1/2/3 modalities: {stage_counts[0]}/{stage_counts[1]}/{stage_counts[2]}")
# ===============================================================
#  Visualization code: Sensitivity analysis (Heatmap - highly recommended)
# ===============================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  
import os

# Ensure there is data
if 'results' in locals() and len(results) > 0:
    print("\nGenerating heatmap...")
    dataset_name = os.path.basename(base_path)

    # 1. Convert data to Pandas DataFrame
    # Your results store [c2, c3, avg_stage]
    df = pd.DataFrame(results, columns=['Cost of 2nd Modality (c2)', 'Cost of 3rd Modality (c3)', 'Avg Modalities'])

    # 2. Convert to matrix form (Pivot)
    # Vertical axis is c2, horizontal axis is c3, middle value is Avg Modalities
    heatmap_data = df.pivot(index='Cost of 2nd Modality (c2)',
                            columns='Cost of 3rd Modality (c3)',
                            values='Avg Modalities')

    # The index here defaults to smallest to largest. We may wish c2=0 at the top or bottom.
    # By default usually the bottom up drawing, since pivot auto-sorts, 0 will be at the top

    # 3. Set up canvas
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")

    # 4. Plot heatmap
    # cmap="rocket_r": This is a very advanced color scheme.
    # "_r" means reverse, which means the larger the value (more modalities), the darker the color, the smaller the value (fewer modalities), the lighter the color
    # You can also switch to "YlGnBu_r" (blue-green) or "Blues_r"
    ax = sns.heatmap(heatmap_data,
                     annot=True,  # Show specific numbers in cells
                     fmt=".2f",  # Keep two decimal places
                     cmap="YlGnBu",  # Color scheme: smaller values lighter (blue), larger values darker
                     linewidths=1,  # Add white lines between cells
                     linecolor='white',
                     cbar_kws={'label': 'Average Number of Modalities Used'},
                     annot_kws={"size": 12, "weight": "bold"})  # Bold font for numbers

    # 5. Adjust axes
    plt.title(f'{dataset_name}: Sensitivity of Modality Usage to Costs', fontsize=16, pad=20, weight='bold')
    plt.xlabel('Cost of 3rd Modality (c3)', fontsize=14, weight='bold')
    plt.ylabel('Cost of 2nd Modality (c2)', fontsize=14, weight='bold')

    # Adjust tick font
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)  # Make vertical axis labels horizontal for easy reading

    # 6. Save
    plt.tight_layout()
    save_name = f"heatmap_{dataset_name}.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap generated: {save_name}")
    plt.show()

else:
    print("❌ Error: results list is empty.")

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import Counter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Current device:", device, "| CUDA available:", torch.cuda.is_available())

# ===== Speed optimization: Low-level library optimization ====="
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

base_path = r"./KIPAN"
GAMMA = 1.0  # No discounting
EPOCHS = 950
BATCH_SIZE = 1024
N_STAGE = 3  # 3 modalities


# ===============================================================
# Main training function
# ===============================================================
def train_once(COST_MAP):
    # ---------- Data ----------
    def load_feat(feat_csv, data_csv):
        feat = pd.read_csv(feat_csv, header=None).squeeze().tolist()
        data = pd.read_csv(data_csv, header=None).dropna(axis=1)
        if len(feat) < data.shape[1]:
            data = data.iloc[:, :len(feat)]
        data.columns = feat[:data.shape[1]]
        return data.values.astype(np.float32)

    X1 = load_feat(f"{base_path}/1_featname.csv", f"{base_path}/1_tr.csv")
    X2 = load_feat(f"{base_path}/2_featname.csv", f"{base_path}/2_tr.csv")
    X3 = load_feat(f"{base_path}/3_featname.csv", f"{base_path}/3_tr.csv")

    X1 = torch.tensor(X1, device=device)
    X2 = torch.tensor(X2, device=device)
    X3 = torch.tensor(X3, device=device)

    labels = pd.read_csv(f"{base_path}/labels_tr.csv", header=None).squeeze().astype(np.int64).values
    labels = torch.tensor(labels, device=device)
    n_class = int(labels.max()) + 1
    n_samples = labels.shape[0]

    stage_dims = [X1.shape[1], X2.shape[1], X3.shape[1]]

    # ---------- Network ----------
    class StageNet(nn.Module):
        def __init__(self, stage_dims, n_class):
            super().__init__()
            self.encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU()
                ) for d in stage_dims
            ])
            fusion_dim = 32 * len(stage_dims)
            self.backbone = nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            # 🎯 3-dimensional action space: [0:Stop, 1:Get X1, 2:Get X3]
            # (Because X2 is mandatory, no need to purchase)
            self.q_head = nn.Linear(64, 3)
            self.clf_head = nn.Linear(64, n_class)

        def forward(self, xs):
            embs = [enc(x) for enc, x in zip(self.encoders, xs)]
            x = torch.cat(embs, dim=1)
            z = self.backbone(x)
            q = self.q_head(z)
            log = self.clf_head(z)
            return q, log

    nets = [StageNet(stage_dims, n_class).to(device) for _ in range(N_STAGE)]
    optims = [optim.Adam(nets[s].parameters(), lr=1e-4, weight_decay=1e-3) for s in range(N_STAGE)]
    class_counts = torch.bincount(labels)
    class_weights = n_samples / (n_class * class_counts.float())
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    mse_loss = nn.MSELoss()

    # ---------- Utilities ----------
    def build_state_by_mask(mask, i):
        states = []
        for used, X in zip(mask, [X1, X2, X3]):
            if used == 1:
                states.append(X[i].unsqueeze(0))
            else:
                states.append(torch.zeros((1, X.shape[1]), dtype=torch.float32, device=device))
        return states

    # 🎯 Force X1 (index 0) to always be in initial state
    def sample_mask(stage, n_stage=3):
        import random
        inds = [1, 2]  # Candidate pool becomes X2 and X3
        chosen = random.sample(inds, stage)
        mask = [1, 0, 0]  # X1 always set to 1
        for c in chosen:
            mask[c] = 1
        return mask

    writer = SummaryWriter('BRCArecord')

    # ---------- Backward recursive training ----------
    for stage in reversed(range(N_STAGE)):
        print(f"\n====== Training Stage {stage + 1} ======")
        for j in range(stage + 1, N_STAGE):
            nets[j].eval()
        nets[stage].train()

        for epoch in range(EPOCHS):
            epoch_q_loss = 0.0
            epoch_clf_loss = 0.0
            perm = torch.randperm(n_samples, device=device)

            for batch_start in range(0, n_samples, BATCH_SIZE):
                batch_inds = perm[batch_start:batch_start + BATCH_SIZE]
                batch_masks = [sample_mask(stage, n_stage=N_STAGE) for _ in batch_inds]

                mask_tensor = torch.tensor(batch_masks, device=device, dtype=X1.dtype)
                X_list = [X1, X2, X3]
                batch_states = []
                for m in range(N_STAGE):
                    feats = X_list[m][batch_inds, :]
                    feats = feats * mask_tensor[:, m:m + 1]
                    batch_states.append(feats)

                target = labels[batch_inds]
                q, logits = nets[stage](batch_states)

                pred_label = logits.argmax(dim=1)
                is_correct = (pred_label == target).long()
                used_modalities = [sum(mask) for mask in batch_masks]
                stop_reward = torch.tensor(
                    [(1.0 if c else -1.0) - COST_MAP[u] for c, u in zip(is_correct, used_modalities)],
                    device=device
                )

                # 🎯 Target Q value construction
                target_q = q.clone().detach()
                target_q[:, 0] = stop_reward

                if stage < N_STAGE - 1:
                    cand_inputs = {k: [] for k in range(N_STAGE)}
                    cand_indices = {k: [] for k in range(N_STAGE)}
                    cand_masks_list = {k: [] for k in range(N_STAGE)}

                    for bi, i in enumerate(batch_inds):
                        curr_mask = batch_masks[bi]
                        # 🎯 Only iterate X1 (0) and X3 (2)
                        for m_idx in [1, 2]:
                            if curr_mask[m_idx] == 0:
                                cand_mask = curr_mask.copy()
                                cand_mask[m_idx] = 1
                                next_k = sum(cand_mask) - 1
                                cand_inputs[next_k].append(build_state_by_mask(cand_mask, i))

                                # 🎯 Action mapping: m_idx 0 corresponds to action 1 (buy X1), m_idx 2 corresponds to action 2 (buy X3)
                                action_idx = 1 if m_idx == 1 else 2
                                cand_indices[next_k].append((bi, action_idx))
                                cand_masks_list[next_k].append(cand_mask)

                    for next_k, states_list in cand_inputs.items():
                        if len(states_list) == 0: continue

                        merged_states = [torch.cat([st[m] for st in states_list], dim=0).to(device) for m in
                                         range(N_STAGE)]
                        masks_list = cand_masks_list[next_k]

                        with torch.no_grad():
                            next_q, _ = nets[next_k](merged_states)

                            # 🚫 Mask invalid actions before calculating max: action 1 corresponds to X1, action 2 corresponds to X3
                            for idx, c_mask in enumerate(masks_list):
                                if c_mask[1] == 1: next_q[idx, 1] = float('-inf')
                                if c_mask[2] == 1: next_q[idx, 2] = float('-inf')

                            next_maxq = next_q.max(dim=1)[0]

                        # Store into corresponding target
                        for idx, (bi, action_idx) in enumerate(cand_indices[next_k]):
                            target_q[bi, action_idx] = GAMMA * next_maxq[idx]

                # Backpropagation & optimization
                L_q = mse_loss(q, target_q)
                L_clf = ce_loss(logits, target)
                loss = L_q + L_clf

                optims[stage].zero_grad()
                loss.backward()
                optims[stage].step()

                epoch_q_loss += L_q.item()
                epoch_clf_loss += L_clf.item()

            epoch_avg_q_loss = epoch_q_loss / n_samples
            epoch_avg_clf_loss = epoch_clf_loss / n_samples
            epoch_avg_total_loss = epoch_avg_q_loss + epoch_avg_clf_loss

            writer.add_scalar(f'Stage_{stage + 1}/Q_Loss', epoch_avg_q_loss, epoch)
            writer.add_scalar(f'Stage_{stage + 1}/Clf_Loss', epoch_avg_clf_loss, epoch)
            writer.add_scalar(f'Stage_{stage + 1}/Total_Loss', epoch_avg_total_loss, epoch)

            if epoch % 20 == 0:
                print(f"epoch {epoch:3d} | Q loss {epoch_avg_q_loss:.4f} | Clf loss {epoch_avg_clf_loss:.4f}")

        nets[stage].eval()

    writer.close()

    # ---------- Save ----------
    save_dir = rf"./KIPAN/14"
    os.makedirs(save_dir, exist_ok=True)

    for s in range(N_STAGE):
        model_filename = f"StageNet_c2{COST_MAP[2]}_c3{COST_MAP[3]}_stage{s + 1}.pth"
        torch.save(nets[s].state_dict(), os.path.join(save_dir, model_filename))
    print(f"Training complete, models saved to {save_dir}")

    # ---------- Inference and dynamic evaluation (test set blind guess) ----------
    for s in range(N_STAGE):
        model_filename = f"StageNet_c2{COST_MAP[2]}_c3{COST_MAP[3]}_stage{s + 1}.pth"
        nets[s].load_state_dict(
            torch.load(os.path.join(save_dir, model_filename), map_location=device))
        nets[s].eval()

    used_stage = []
    correct_dynamic = 0

    with torch.no_grad():
        for i in range(n_samples):
            # 🎯 Always start from X2
            mask = [1, 0, 0]
            while True:
                curr_k = sum(mask) - 1
                st = build_state_by_mask(mask, i)
                q, logits = nets[curr_k](st)

                q_values = q.squeeze()

                # 🎯 Mask already unlocked modalities: if X1 exists mask action 1, if X3 exists mask action 2
                if mask[1] == 1: q_values[1] = float('-inf')
                if mask[2] == 1: q_values[2] = float('-inf')

                action = q_values.argmax().item()

                if action == 0 or sum(mask) == N_STAGE:
                    used_stage.append(sum(mask))
                    pred_label = logits.argmax(dim=1).item()
                    if pred_label == labels[i].item():
                        correct_dynamic += 1
                    break
                else:
                    # 🎯 Map action to actual data unlock
                    if action == 1:
                        mask[1] = 1  # Unlock X1
                    elif action == 2:
                        mask[2] = 1  # Unlock X3

    stat = Counter(used_stage)
    print("[Training Set] Distribution of early stopping by strategy:")
    for s in range(1, N_STAGE + 1):
        print(f" Stopped at {s} modality: {stat[s]} samples ({stat[s] / n_samples:.2%})")

    accuracy = correct_dynamic / n_samples
    return accuracy, epoch_avg_total_loss


# ===============================================================
# Main entry point
# ===============================================================
if __name__ == "__main__":
    c2_candidates = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    c3_candidates = [0.2, 0.3, 0.4, 0.5, 0.6]

    for c2 in c2_candidates:
        for c3 in c3_candidates:
            cost_map = {1: 0.0, 2: c2, 3: c3}
            print(f"\n>>> Trying COST_MAP = {cost_map}")
            # Removed the cheating logic of picking best, purely train and save
            train_once(cost_map)

    print("\n✅ Training for all cost combinations is complete! All models have been saved.")
# SDM-Q Multimodal Reinforcement Learning Classification Project

## Overview

SDM-Q is a reinforcement learning classification project for multimodal biomedical data. The project uses three modalities to build sample states. For the **current sample**, the model decides whether the already selected modalities are sufficient for classification. If they are sufficient, the model stops and outputs the prediction. If they are not sufficient, the model adaptively selects the next modality from the remaining unselected modalities.

Therefore, this project does not simply use all modalities for every sample, and it does not only add modalities in a fixed order. Instead, it dynamically chooses between “stop and predict” and “continue by selecting a remaining modality” according to the current state of each sample, balancing classification performance and modality usage cost.

## File Description

| File or Directory | Description |
| --- | --- |
| `1.py` | Training script for the multimodal reinforcement learning classifier |
| `2.py` | Testing/evaluation script for loading trained models and evaluating performance on the test set |
| `requirements.txt` | Python dependency list |
| `run_train.ps1` | Windows PowerShell launcher for training |
| `run_eval.ps1` | Windows PowerShell launcher for testing/evaluation |
| `DataSets/DataSets/` | Dataset directory |
| `BRCA记录/` | TensorBoard logs and training records |

## Dataset Structure

The current project contains four datasets:

```text
DataSets/DataSets/
├── BRCA/
├── KIPAN/
├── LGG/
└── ROSMAP/
```

Each dataset directory contains data files for three modalities:

```text
dataset_name/
├── 1_featname.csv
├── 1_tr.csv
├── 1_te.csv
├── 2_featname.csv
├── 2_tr.csv
├── 2_te.csv
├── 3_featname.csv
├── 3_tr.csv
├── 3_te.csv
├── labels_tr.csv
└── labels_te.csv
```

File descriptions:

- `1_tr.csv`, `2_tr.csv`, `3_tr.csv`: three-modality features for the training set.
- `1_te.csv`, `2_te.csv`, `3_te.csv`: three-modality features for the test set.
- `1_featname.csv`, `2_featname.csv`, `3_featname.csv`: feature names for the corresponding modalities.
- `labels_tr.csv`: training labels.
- `labels_te.csv`: test labels.

## Environment

Recommended environment:

- Windows
- PowerShell
- Python 3.9
- NVIDIA GPU
- CUDA 12.8 compatible driver

Main dependencies:

```text
torch==2.7.1+cu128
pandas
numpy
scikit-learn
seaborn
matplotlib
tensorboard
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Training

`1.py` is the training script. Current main parameters include:

```python
N_STAGE = 3
EPOCHS = 950
BATCH_SIZE = 1024
```

The training script searches different modality cost combinations:

```python
c2_candidates = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
c3_candidates = [0.2, 0.3, 0.4, 0.5, 0.6]
```

Here:

- `c2` is the usage cost of the 2nd modality.
- `c3` is the usage cost of the 3rd modality.
- Higher cost makes the model more likely to stop early.
- Lower cost makes the model more likely to use more modalities.

Run training:

```powershell
.\run_train.ps1
```

Or run directly:

```powershell
C:\ProgramData\Anaconda3\envs\py39\python.exe 1.py
```

After training, model weight files are generated:

```text
StageNet_c2{c2}_c3{c3}_stage1.pth
StageNet_c2{c2}_c3{c3}_stage2.pth
StageNet_c2{c2}_c3{c3}_stage3.pth
```

## Adaptive Modality Selection for the Current Sample

The model does not use one fixed modality combination for all samples. Instead, it makes decisions separately for each current sample. For a sample, the model builds a state from the modalities already used and outputs action values to decide the next step:

- If the current information is sufficient, the model stops and outputs the classification result.
- If the current information is insufficient, the model selects one modality from the remaining unselected modalities.
- After a new modality is added, the model updates the sample state and decides again whether to stop or continue.

The core process is:

```text
current sample state
→ decide whether to stop
→ if not stopping, select one remaining unselected modality
→ update the sample state
→ decide again
```

This allows the model to stop early for easier samples and use more modalities for harder samples, balancing classification performance and modality cost.

## Initial Modality Selection

The current code uses the 1st modality as the initial modality by default. When each sample enters the model, the initial state already contains the 1st modality. The mask is:

```python
mask = [1, 0, 0]
```

Mask meaning:

- The 1st position represents the 1st modality. `1` means selected.
- The 2nd position represents the 2nd modality. `0` means unselected.
- The 3rd position represents the 3rd modality. `0` means unselected.

Therefore, the current default process is:

```text
start from the 1st modality
→ the model decides whether to stop
→ if not stopping, it adaptively selects from the 2nd and 3rd modalities
→ then it decides again whether to stop or continue selecting remaining modalities
```

## Where to Modify the Initial Modality

If you need to change the initial modality, the training and testing scripts must stay consistent. Check the following locations:

| File | Code Location | Purpose |
| --- | --- | --- |
| `1.py` | `mask = [1, 0, 0]` inside `sample_mask()` | Controls the initial modality during training |
| `1.py` | `mask = [1, 0, 0]` in the dynamic evaluation section | Controls the initial modality when reporting policy behavior after training |
| `2.py` | `mask = [1, 0, 0]` inside the inference loop | Controls the initial modality during testing/evaluation |

Common masks:

```text
[1, 0, 0] means starting from the 1st modality
[0, 1, 0] means starting from the 2nd modality
[0, 0, 1] means starting from the 3rd modality
```

For example, to start from the 2nd modality, change the locations above to:

```python
mask = [0, 1, 0]
```

You also need to update the candidate pool in `sample_mask()` in `1.py`. When the 1st modality is used as the initial modality, the remaining candidates are the 2nd and 3rd modalities:

```python
inds = [1, 2]
```

If the 2nd modality is used as the initial modality, the remaining candidates should be the 1st and 3rd modalities:

```python
inds = [0, 2]
```

If the 3rd modality is used as the initial modality, the remaining candidates should be the 1st and 2nd modalities:

```python
inds = [0, 1]
```

Note: if you change the initial modality, you must also check the action mapping logic so that the model output actions correctly correspond to “stop” or “select a remaining modality”. The initial modality, candidate pool, and action meanings must be consistent between training and testing. Otherwise, the testing state distribution will differ from the training state distribution, and evaluation results will be unreliable.

## Testing and Evaluation

`2.py` is the testing/evaluation script. It loads the trained `StageNet_*.pth` weight files and evaluates the model on the specified dataset.

Current default settings in `2.py`:

```python
N_STAGE = 3
USE_TRAIN = False
```

Where:

- `USE_TRAIN = False` evaluates the test set.
- `USE_TRAIN = True` evaluates the training set.

Run testing/evaluation:

```powershell
.\run_eval.ps1
```

Or run directly:

```powershell
C:\ProgramData\Anaconda3\envs\py39\python.exe 2.py
```

Before running `2.py`, the project root must contain the corresponding `StageNet_*.pth` model weight files. If no weight files exist, run `1.py` first.

## Dataset Path

It is recommended that the data paths in both the training and testing scripts point to the dataset directory under the current workspace:

```python
base_path = os.path.join(script_dir, "DataSets", "DataSets", "KIPAN")
```

To switch datasets, replace the final name with one of:

```text
BRCA
KIPAN
LGG
ROSMAP
```

For example, to switch the training dataset to `LGG`:

```python
base_path = os.path.join(script_dir, "DataSets", "DataSets", "LGG")
```

## Outputs

Training outputs:

- training logs for each cost setting,
- Q loss, classification loss, and total loss for each stage,
- modality usage statistics on the training set,
- `StageNet_*.pth` model weight files.

Testing outputs:

- Accuracy,
- Weighted F1,
- Macro-F1,
- AUC for binary classification,
- average number of modalities used,
- number of samples stopped at each stage,
- heatmap files such as `heatmap_ROSMAP.png`.

## TensorBoard

The training script records training curves with `SummaryWriter`. Use TensorBoard to inspect logs:

```powershell
C:\ProgramData\Anaconda3\envs\py39\Scripts\tensorboard.exe --logdir "BRCA记录/第一模态output/runs"
```

## FAQ

### 1. What should I do if model files cannot be found during testing?

Run the training script first to generate `StageNet_*.pth`:

```powershell
.\run_train.ps1
```

Or place existing model weights in the project root.

### 2. What should I do if CUDA memory is insufficient?

Reduce the batch size in `1.py`:

```python
BATCH_SIZE = 512
```

If needed, reduce it further to `256` or `128`.

### 3. What should I do if training takes too long?

Reduce the number of epochs:

```python
EPOCHS = 300
```

You can also reduce the number of candidate values in `c2_candidates` and `c3_candidates`.

### 4. How do I check whether GPU is available?

Run:

```powershell
C:\ProgramData\Anaconda3\envs\py39\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If it outputs `True` and displays the GPU name, the CUDA GPU is available.

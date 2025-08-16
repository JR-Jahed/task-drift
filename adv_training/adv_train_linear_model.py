import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataset import ActivationsDatasetDynamic
from utils.load_file_paths import load_file_paths
from pgd import pgd_torch_linear
from constants import PROJECT_ROOT
import time
from logistic_regression import LogisticRegression



MODEL = 'phi3'
OUTPUT_DIR = f'{PROJECT_ROOT}/adv_trained_linear_probes/{MODEL}'
os.makedirs(OUTPUT_DIR, exist_ok=True)



def make_diffs_from_dataset(dataset):
    """
    dataset yields (primary, clean, poisoned) tensors.
    Returns tensors:
      X_clean_diff: [N, D]   where diff = (clean - primary)
      X_poison_diff: [N, D]  where diff = (poisoned - primary)
    """
    clean_diff, poison_diff = [], []

    for primary, clean, poisoned in tqdm(dataset):

        primary = primary.flatten().float()
        clean = clean.flatten().float()
        poisoned = poisoned.flatten().float()

        clean_diff.append((clean - primary).unsqueeze(0))
        poison_diff.append((poisoned - primary).unsqueeze(0))

    X_clean_diff = torch.cat(clean_diff,  dim=0) if clean_diff else torch.empty(0)
    X_poison_diff = torch.cat(poison_diff, dim=0) if poison_diff else torch.empty(0)

    return X_clean_diff, X_poison_diff


def train_model_pt(
    train_files,
    num_layers,                 # (start_layer, end_layer)
    root_dir,
    input_dim,                  # 3072 for Phi / 4096 for LLaMA
    batch_files=10,
    epochs_per_chunk=10,
    epsilon=0.1,
    alpha=0.01,
    steps=20,
    adv_ratio=1.0,              # how many adv poisoned per clean/poisoned (1.0 means equal count to poisoned)
    lr=1e-3,
    weight_decay=0.0,           # weight_decay ~= L2 regularization
):
    device = 'cpu'
    model = LogisticRegression(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    for i in range(0, len(train_files), batch_files):
        batch_list = train_files[i:i+batch_files]

        # Load a chunk of data
        dataset = ActivationsDatasetDynamic(
            batch_list,
            root_dir=root_dir,
            num_layers=num_layers
        )

        # Build clean/poison diffs
        X_clean_diff, X_poison_diff = make_diffs_from_dataset(dataset)
        if X_clean_diff.numel() == 0:
            continue

        # Create adversarial poisoned examples *on-the-fly* against the current model
        # We need the original (primary, poisoned) to craft adv; rebuild a simple loop:
        adv_poison_list = []
        for primary, clean, poisoned in tqdm(dataset, desc="PGD crafting"):
            primary  = primary.flatten().float().to(device)
            poisoned = poisoned.flatten().float().to(device)
            adv_poisoned = pgd_torch_linear(
                model,
                primary=primary,
                poisoned=poisoned,
                epsilon=epsilon,
                alpha=alpha,
                steps=steps,
                target_label=0
            )
            adv_poison_list.append(((adv_poisoned - primary).unsqueeze(0)).cpu())

        X_adv_poison_diff = torch.cat(adv_poison_list, dim=0) if adv_poison_list else torch.empty(0)

        # Labels: clean = 0, poisoned = 1, adv_poisoned = 1
        y_clean = torch.zeros(X_clean_diff.size(0), dtype=torch.float32)
        y_poison = torch.ones(X_poison_diff.size(0), dtype=torch.float32)
        y_adv = torch.ones(X_adv_poison_diff.size(0), dtype=torch.float32)

        # Optionally subsample adv to control ratio
        if adv_ratio is not None and adv_ratio != 1.0 and X_adv_poison_diff.size(0) > 0:
            k = int(min(X_adv_poison_diff.size(0), adv_ratio * X_poison_diff.size(0)))
            idx = torch.randperm(X_adv_poison_diff.size(0))[:k]
            X_adv_poison_diff = X_adv_poison_diff[idx]
            y_adv = y_adv[idx]

        # Concatenate and shuffle within the chunk
        X_chunk = torch.cat([X_clean_diff, X_poison_diff, X_adv_poison_diff], dim=0)
        y_chunk = torch.cat([y_clean,      y_poison,      y_adv],            dim=0)

        perm = torch.randperm(X_chunk.size(0))
        X_chunk = X_chunk[perm]
        y_chunk = y_chunk[perm]

        # DataLoader for minibatching
        ds = TensorDataset(X_chunk, y_chunk)
        loader = DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)

        # Train for a few epochs on this chunk
        for _ in range(epochs_per_chunk):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)              # [B]
                loss = criterion(logits, yb)    # BCE-with-logits

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return model


if __name__ == "__main__":

    filepaths = load_file_paths(f'{PROJECT_ROOT}/data_files/train_files_{MODEL}.txt')

    num_layer = 23
    os.makedirs(os.path.join(OUTPUT_DIR, str(num_layer)), exist_ok=True)
    layer_output_dir = os.path.join(OUTPUT_DIR, str(num_layer))

    linear_model = pickle.load(open(os.path.join(f'{PROJECT_ROOT}/trained_linear_probes_microsoft',
                                                 MODEL, str(num_layer), 'model.pickle'), 'rb'))

    input_dim = linear_model.coef_.shape[1]
    num_layers = (num_layer, num_layer)
    root_dir = '/mnt/6052137152134B64/Activation/phi__3__3.8/training'

    start = time.time()

    model = train_model_pt(
        train_files=filepaths,
        num_layers=num_layers,
        root_dir=root_dir,
        input_dim=input_dim,
        batch_files=10,
        epochs_per_chunk=10,
        epsilon=0.5,
        alpha=0.01,
        steps=20,
        adv_ratio=1.0,
        lr=1e-3,
        weight_decay=1e-4,
    )

    end = time.time()

    print(f"Training took {end - start} seconds")

    torch.save(model.state_dict(), os.path.join(layer_output_dir, 'model.pt'))

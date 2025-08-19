import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataset import ActivationsDatasetDynamic, ActivationsDatasetDynamicPrimaryText
from utils.load_file_paths import load_file_paths
from pgd import pgd_torch_linear
from constants import PROJECT_ROOT, ROOT_DIR_TRAIN, ROOT_DIR_VAL
import time
from logistic_regression import LogisticRegression
import random
import matplotlib.pyplot as plt

MODEL = 'phi3'
OUTPUT_DIR = f'{PROJECT_ROOT}/adv_trained_linear_probes/{MODEL}'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def prepare_val_data(
        val_clean_files,
        val_poisoned_files,
        root_dir_val,
        num_layers,
        model,
        epsilon,
        alpha,
        steps
):
    val_clean_dataset = ActivationsDatasetDynamicPrimaryText(
        val_clean_files,
        root_dir=root_dir_val,
        num_layers=num_layers,
    )
    val_poisoned_dataset = ActivationsDatasetDynamicPrimaryText(
        val_poisoned_files,
        root_dir=root_dir_val,
        num_layers=num_layers,
    )

    clean_diff, poisoned_diff, adv_poisoned_diff = [], [], []

    for primary, clean in tqdm(val_clean_dataset):
        primary = primary.flatten().float()
        clean = clean.flatten().float()

        clean_diff.append((clean - primary).unsqueeze(0))

    for primary, poisoned in tqdm(val_poisoned_dataset):
        primary = primary.flatten().float()
        poisoned = poisoned.flatten().float()

        poisoned_diff.append((poisoned - primary).unsqueeze(0))

        adv_poisoned = pgd_torch_linear(
            model,
            primary=primary,
            poisoned=poisoned,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            target_label=0
        )
        adv_poisoned_diff.append(((adv_poisoned - primary).unsqueeze(0)).cpu())

    X_clean_diff = torch.cat(clean_diff, dim=0)
    X_poisoned_diff = torch.cat(poisoned_diff, dim=0)
    X_adv_poisoned_diff = torch.cat(adv_poisoned_diff, dim=0)

    y_clean = torch.zeros(X_clean_diff.size(0), dtype=torch.float32)
    y_poisoned = torch.ones(X_poisoned_diff.size(0), dtype=torch.float32)
    y_adv = torch.ones(X_adv_poisoned_diff.size(0), dtype=torch.float32)

    X_chunk = torch.cat([X_clean_diff, X_poisoned_diff, X_adv_poisoned_diff], dim=0)
    y_chunk = torch.cat([y_clean, y_poisoned, y_adv], dim=0)

    ds = TensorDataset(X_chunk, y_chunk)
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    return loader


def make_diffs_from_dataset(train_dataset):
    """
    train_dataset yields (primary, clean, poisoned) tensors.
    Returns tensors:
      X_clean_diff: [N, D]   where diff = (clean - primary)
      X_poisoned_diff: [N, D]  where diff = (poisoned - primary)
    """
    clean_diff, poisoned_diff = [], []

    for primary, clean, poisoned in tqdm(train_dataset):

        primary = primary.flatten().float()
        clean = clean.flatten().float()
        poisoned = poisoned.flatten().float()

        clean_diff.append((clean - primary).unsqueeze(0))
        poisoned_diff.append((poisoned - primary).unsqueeze(0))

    X_clean_diff = torch.cat(clean_diff,  dim=0)
    X_poisoned_diff = torch.cat(poisoned_diff, dim=0)

    return X_clean_diff, X_poisoned_diff


def train_model_pt(
    train_files,
    val_clean_files,
    val_poisoned_files,
    num_layers,                 # (start_layer, end_layer)
    root_dir_train,
    root_dir_val,
    input_dim,                  # 3072 for Phi / 4096 for LLaMA
    batch_files=10,
    epochs_per_chunk=3,
    epsilon=0.1,
    alpha=0.01,
    steps=20,
    adv_ratio=1.0,              # how many adv poisoned per clean/poisoned (1.0 means equal count to poisoned)
    lr=1e-3,
    weight_decay=0.0,           # weight_decay ~= L2 regularization
    validate_every=1
):
    device = 'cpu'
    model = LogisticRegression(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    for i in range(0, len(train_files), batch_files):
        print(f"Currently processing: {i}-{i+batch_files}")
        batch_list = train_files[i:i+batch_files]

        # Load a chunk of data
        train_dataset = ActivationsDatasetDynamic(
            batch_list,
            root_dir=root_dir_train,
            num_layers=num_layers
        )
        if len(train_dataset) == 0:
            continue

        # Build clean/poison diffs
        X_clean_diff, X_poisoned_diff = make_diffs_from_dataset(train_dataset)

        # Create adversarial poisoned examples *on-the-fly* against the current model
        # We need the original (primary, poisoned) to craft adv; rebuild a simple loop:
        adv_poisoned_diff = []
        for primary, clean, poisoned in tqdm(train_dataset, desc="PGD crafting"):
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
            adv_poisoned_diff.append(((adv_poisoned - primary).unsqueeze(0)).cpu())

        X_adv_poisoned_diff = torch.cat(adv_poisoned_diff, dim=0)

        # Labels: clean = 0, poisoned = 1, adv_poisoned = 1
        y_clean = torch.zeros(X_clean_diff.size(0), dtype=torch.float32)
        y_poisoned = torch.ones(X_poisoned_diff.size(0), dtype=torch.float32)
        y_adv = torch.ones(X_adv_poisoned_diff.size(0), dtype=torch.float32)

        # Optionally subsample adv to control ratio
        if adv_ratio is not None and adv_ratio != 1.0 and X_adv_poisoned_diff.size(0) > 0:
            k = int(min(X_adv_poisoned_diff.size(0), adv_ratio * X_poisoned_diff.size(0)))
            idx = torch.randperm(X_adv_poisoned_diff.size(0))[:k]
            X_adv_poisoned_diff = X_adv_poisoned_diff[idx]
            y_adv = y_adv[idx]

        # Concatenate and shuffle within the chunk
        X_chunk = torch.cat([X_clean_diff, X_poisoned_diff, X_adv_poisoned_diff], dim=0)
        y_chunk = torch.cat([y_clean,      y_poisoned,      y_adv],            dim=0)

        perm = torch.randperm(X_chunk.size(0))
        X_chunk = X_chunk[perm]
        y_chunk = y_chunk[perm]

        # DataLoader for minibatching
        ds = TensorDataset(X_chunk, y_chunk)
        loader = DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)

        val_loader = prepare_val_data(
            val_clean_files=random.sample(val_clean_files, 2),
            val_poisoned_files=random.sample(val_poisoned_files, 2),
            root_dir_val=root_dir_val,
            num_layers=num_layers,
            model=model,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
        )

        best_val_acc = 0.0
        patience = 3
        patience_counter = 0

        stop_training = False

        # Train for a few epochs on this chunk
        for epoch in range(epochs_per_chunk):
            if stop_training:
                break

            print(f"Epoch {epoch+1}/{epochs_per_chunk}")

            for step, (xb, yb) in enumerate(loader):
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)              # [B]
                loss = criterion(logits, yb)    # BCE-with-logits

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step > 0 and step % validate_every == 0:
                    model.eval()
                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for xval, yval in val_loader:
                            xval = xval.to(device)
                            yval = yval.to(device)

                            out = model(xval)
                            preds = (torch.sigmoid(out) > 0.5).long()

                            correct += (preds.squeeze() == yval.long()).sum().item()
                            total += yval.size(0)

                    model.train()
                    val_acc = correct / total

                    print(f"Step: {step}   Validation accuracy: {val_acc * 100:.2f}%")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print("Early stopping triggered")
                            stop_training = True
                            break


    return model


if __name__ == "__main__":

    train_filepaths = load_file_paths(f'{PROJECT_ROOT}/data_files/train_files_{MODEL}.txt')
    val_clean_filepaths = load_file_paths(f'{PROJECT_ROOT}/data_files/val_clean_files_{MODEL}.txt')
    val_poisoned_filepaths = load_file_paths(f'{PROJECT_ROOT}/data_files/val_poisoned_files_{MODEL}.txt')

    random.shuffle(train_filepaths)

    num_layer = 31
    os.makedirs(os.path.join(OUTPUT_DIR, str(num_layer)), exist_ok=True)
    layer_output_dir = os.path.join(OUTPUT_DIR, str(num_layer))

    linear_model = pickle.load(open(os.path.join(f'{PROJECT_ROOT}/trained_linear_probes_microsoft',
                                                 MODEL, str(num_layer), 'model.pickle'), 'rb'))

    input_dim = linear_model.coef_.shape[1]
    num_layers = (num_layer, num_layer)
    root_dir_train = ROOT_DIR_TRAIN[MODEL]
    root_dir_val = ROOT_DIR_VAL[MODEL]

    start = time.time()

    model = train_model_pt(
        train_files=train_filepaths,
        val_clean_files=val_clean_filepaths,
        val_poisoned_files=val_poisoned_filepaths,
        num_layers=num_layers,
        root_dir_train=root_dir_train,
        root_dir_val=root_dir_val,
        input_dim=input_dim,
        batch_files=8,
        epochs_per_chunk=20,
        epsilon=0.5,
        alpha=0.01,
        steps=20,
        adv_ratio=1.0,
        lr=1e-3,
        weight_decay=1e-4,
        validate_every=2,
    )

    end = time.time()

    print(f"Training took {end - start} seconds")

    torch.save(model.state_dict(), os.path.join(layer_output_dir, 'model.pt'))

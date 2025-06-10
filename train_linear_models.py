import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import pickle
from load_file_paths import load_file_paths
from dataset import ActivationsDatasetDynamic


MODEL = "phi3"
OUTPUT_DIR = MODEL
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Which layers would be used for training probes
LAYERS_PER_MODEL = {
    'phi3': [0, 7, 15, 23, 31],
    'llama3_8b': [0, 7, 15, 23, 31],
}


def train_model(train_files, num_layers):
    print("Loading dataset.")

    dataset = ActivationsDatasetDynamic(train_files, root_dir='/mnt/6052137152134B64/Activation/phi__3__3.8/training', num_layers=num_layers)

    print("Processing dataset.")

    clean_diff = []
    poisoned_diff = []

    for primary, clean, poisoned in tqdm(dataset):
        clean_diff.append((clean - primary).flatten().float().numpy())
        poisoned_diff.append((poisoned - primary).flatten().float().numpy())

    y = [0] * len(dataset) + [1] * len(dataset)
    X = clean_diff + poisoned_diff

    print("Training logistic regression classifier.")

    model = LogisticRegression()
    model.fit(X, y)

    return model


if __name__ == "__main__":

    LAYERS = LAYERS_PER_MODEL[MODEL]

    for n_layer in LAYERS:
        print(f"[*] Training model for the {n_layer}-th activation layer.")
        os.makedirs(os.path.join(OUTPUT_DIR, str(n_layer)), exist_ok=True)
        layer_output_dir = os.path.join(OUTPUT_DIR, str(n_layer))

        # Train the model.
        train_files = load_file_paths('data_files/train_files_phi3.txt')

        print(f"Training model with {len(train_files)} files.")

        # Train the linear model on a small subset of activations
        model = train_model(train_files[:5], num_layers=(n_layer, n_layer))
        pickle.dump(model, open(os.path.join(layer_output_dir, 'model.pickle'), "wb"))

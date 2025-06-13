import os
import sys

import numpy as np
import torch
from fgsm import fgsm
from pgd import pgd
import pickle

from dataset import ActivationsDatasetDynamicPrimaryText
from load_file_paths import load_file_paths

model = 'llama3_8b'

models = {
    'phi3': 'phi__3__3.8',
    'llama3_8b': 'llama__3__8',
}

ROOT_DIR = {
    'phi3': '/mnt/12EA576EEA574D5B/Activation/phi__3__3.8/test',
    'llama3_8b': '/mnt/12EA576EEA574D5B/Activation/llama__3__8B/test'
}

LAYERS_PER_MODEL = {
    'phi3': [0, 7, 15, 23, 31],
    'llama3_8b': [0, 7, 15, 23, 31],
}


def get_activations(filepath, num_layer, linear_model, confidence=-1):

    dataset = ActivationsDatasetDynamicPrimaryText(
        filepath,
        root_dir=ROOT_DIR[model],
        num_layers=(num_layer, num_layer)
    )

    diff = []

    for primary, text in dataset:
        diff.append((text - primary).flatten().float().numpy())

    predict_proba = linear_model.predict_proba(diff)

    activations = []

    label = 0 if "clean" in filepath[0] else 1

    for i, prob in enumerate(predict_proba):

        if confidence == -1:
            # Select all the correctly classified instances
            if prob[label] > .5:
                activations.append((dataset[i][0].flatten().float().numpy(), dataset[i][1].flatten().float().numpy()))
        else:
            # Select a subset of correctly classified instances where the
            # confidence score is in the range [confidence, confidence + .1)
            if confidence <= prob[label] < confidence + .1:
                activations.append((dataset[i][0].flatten().float().numpy(), dataset[i][1].flatten().float().numpy()))

    activations = np.array(activations)
    return activations


if __name__ == "__main__":

    filepaths = load_file_paths(f'../data_files/test_poisoned_files_{model}.txt')

    layer_index = 4
    num_layer = LAYERS_PER_MODEL[model][layer_index]

    linear_model = pickle.load(open(os.path.join('../trained_linear_probes_microsoft',
                                        model, str(num_layer), 'model.pickle'), 'rb'))

    correctly_classified_instances = 0
    count_success = 0

    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    for i in range(len(filepaths)):

        activations = get_activations(filepaths[i: i + 1], num_layer, linear_model)

        for activation in activations:
            success = fgsm(linear_model, activation, 1, epsilon=.02)
            # success = pgd(linear_model, activation, 1, epsilon=.02, alpha=.001, num_iter=20)
            if success:
                count_success += 1

        correctly_classified_instances += activations.shape[0]

    sys.stdout = original_stdout

    print(f"\nlayer {num_layer}   Total correctly classified instances: {correctly_classified_instances}\nSuccess: {count_success}")

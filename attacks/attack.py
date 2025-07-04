import os
import sys

from fgsm import fgsm
from pgd import pgd
import pickle
import json
import time

from dataset import ActivationsDatasetDynamicPrimaryText
from load_file_paths import load_file_paths
from constants import ROOT_DIR, LAYERS_PER_MODEL

model = 'phi3'


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
    indices = []

    label = 0 if "clean" in filepath[0] else 1

    for i, prob in enumerate(predict_proba):

        if confidence == -1:
            # Select all the correctly classified instances
            if prob[label] > .5:
                activations.append((dataset[i][0].flatten().float().detach(), dataset[i][1].flatten().float().detach()))
                indices.append(i)
        else:
            # Select a subset of correctly classified instances where the
            # confidence score is in the range [confidence, confidence + .1)
            if confidence <= prob[label] < confidence + .1:
                activations.append((dataset[i][0].flatten().float().detach(), dataset[i][1].flatten().float().detach()))
                indices.append(i)

    return activations, indices


if __name__ == "__main__":

    filepaths = load_file_paths(f'../data_files/test_poisoned_files_{model}.txt')

    epsilons = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5]

    attack_type = 'pgd'

    file_path_asr = f'{attack_type}_attack_details_on_{model}.json'
    file_path_indices = f'{attack_type}_indices_of_successful_attacks_on_{model}.json'

    if os.path.exists(file_path_asr):
        with open(file_path_asr, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    if os.path.exists(file_path_indices):
        with open(file_path_indices, 'r') as f:
            data_indices = json.load(f)
    else:
        data_indices = {}

    start = time.time()

    for num_layer in LAYERS_PER_MODEL[model]:

        if str(num_layer) not in data:
            data[str(num_layer)] = {}

        if str(num_layer) not in data_indices:
            data_indices[str(num_layer)] = {}

        for epsilon in epsilons:

            linear_model = pickle.load(open(os.path.join('../trained_linear_probes_microsoft',
                                                model, str(num_layer), 'model.pickle'), 'rb'))

            correctly_classified_instances = 0
            count_success = 0

            # Disable print
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

            if f'epsilon_{epsilon}' not in data_indices[str(num_layer)]:
                data_indices[str(num_layer)][f'epsilon_{epsilon}'] = {}

            for i in range(len(filepaths)):

                activations, indices = get_activations(filepaths[i: i + 1], num_layer, linear_model)
                indices_successful_attack = []

                for index, activation in zip(indices, activations):
                    if attack_type == 'fgsm':
                        success = fgsm(linear_model, activation, 1, epsilon=epsilon)
                    else:
                        success = pgd(linear_model, activation, 1, epsilon=epsilon, alpha=.01, num_iter=20)
                    if success:
                        count_success += 1
                        indices_successful_attack.append(index)

                data_indices[str(num_layer)][f'epsilon_{epsilon}'][filepaths[i]] = indices_successful_attack

                correctly_classified_instances += len(activations)

            success_rate = count_success / correctly_classified_instances

            if 'epsilon' not in data[str(num_layer)]:
                data[str(num_layer)]['epsilon'] = []

            if 'success_rate' not in data[str(num_layer)]:
                data[str(num_layer)]['success_rate'] = []

            data[str(num_layer)]['epsilon'].append(epsilon)
            data[str(num_layer)]['success_rate'].append(success_rate)

            # Enable print again
            sys.stdout = original_stdout

            print(f"\nlayer {num_layer}   Total correctly classified instances: {correctly_classified_instances}\nSuccess: {count_success}")

    end = time.time()
    print("Total time: ", end - start)

    json.dump(data, open(file_path_asr, 'w'))
    json.dump(data_indices, open(file_path_indices, 'w'))

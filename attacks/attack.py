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

model = 'llama3_8b'


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
                activations.append((dataset[i][0].flatten().float().detach(), dataset[i][1].flatten().float().detach()))
        else:
            # Select a subset of correctly classified instances where the
            # confidence score is in the range [confidence, confidence + .1)
            if confidence <= prob[label] < confidence + .1:
                activations.append((dataset[i][0].flatten().float().detach(), dataset[i][1].flatten().float().detach()))

    return activations


if __name__ == "__main__":

    filepaths = load_file_paths(f'../data_files/test_poisoned_files_{model}.txt')

    epsilons = [.005, .01, .02]

    attack_type = 'pgd'

    _map = {}

    start = time.time()

    for num_layer in LAYERS_PER_MODEL[model]:

        if num_layer not in _map:
            _map[num_layer] = {}

        for epsilon in epsilons:

            linear_model = pickle.load(open(os.path.join('../trained_linear_probes_microsoft',
                                                model, str(num_layer), 'model.pickle'), 'rb'))

            correctly_classified_instances = 0
            count_success = 0

            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

            for i in range(len(filepaths)):

                activations = get_activations(filepaths[i: i + 1], num_layer, linear_model)

                for activation in activations:
                    if attack_type == 'fgsm':
                        success = fgsm(linear_model, activation, 1, epsilon=epsilon)
                    else:
                        success = pgd(linear_model, activation, 1, epsilon=epsilon, alpha=.001, num_iter=20)
                    if success:
                        count_success += 1

                correctly_classified_instances += len(activations)

            success_rate = count_success / correctly_classified_instances

            if 'epsilon' not in _map[num_layer]:
                _map[num_layer]['epsilon'] = []

            if 'success_rate' not in _map[num_layer]:
                _map[num_layer]['success_rate'] = []

            _map[num_layer]['epsilon'].append(epsilon)
            _map[num_layer]['success_rate'].append(success_rate)

            sys.stdout = original_stdout

            print(f"\nlayer {num_layer}   Total correctly classified instances: {correctly_classified_instances}\nSuccess: {count_success}")

    end = time.time()
    print("Total time: ", end - start)

    json.dump(_map, open(f'{attack_type}_attack_details_on_{model}.json', 'w'))

import pickle
from dataset import ActivationsDatasetDynamicPrimaryText
from utils.load_file_paths import load_file_paths
import numpy as np
import os
from collections import defaultdict
from constants import PROJECT_ROOT, LAYERS_PER_MODEL, ROOT_DIR_TEST
import torch
from adv_training.logistic_regression import LogisticRegression


np.set_printoptions(suppress=True, linewidth=10000)

model = 'phi3'


def test_adv_trained_model(test_files_path, num_layer):

    test_files = load_file_paths(test_files_path)

    linear_model = LogisticRegression(input_dim=3072)
    linear_model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'adv_trained_linear_probes_suffix', model, str(num_layer), 'model.pt')))

    linear_model.eval()

    total_prompts = 31134
    poisoned_predicted = 0

    batch_files = 1

    for i in range(0, len(test_files), batch_files):

        # Test the linear model on a small subset of activations
        dataset = ActivationsDatasetDynamicPrimaryText(
            test_files[i: i + batch_files],
            root_dir=ROOT_DIR_TEST[model],
            num_layers=(num_layer, num_layer)
        )

        diff = []

        for primary, text in dataset:
            diff.append((text - primary).flatten().float())

        diff = torch.stack(diff)

        with torch.no_grad():
            logits = linear_model(diff)
            probs = torch.sigmoid(logits)

        predictions = (probs >= .5).long()
        poisoned_predicted += predictions.count_nonzero().item()

    if "clean" in test_files_path:
        print(f"Correct: {total_prompts - poisoned_predicted}  Test accuracy: {(total_prompts - poisoned_predicted) / total_prompts * 100:.2f}%")
    else:
        print(f"Correct: {poisoned_predicted}  Test accuracy: {poisoned_predicted / total_prompts * 100:.2f}%")


def test_microsoft_trained_model(test_files_path, num_layer):

    test_files = load_file_paths(test_files_path)

    total_prompts = 31134
    poisoned_predicted = 0

    linear_model = pickle.load(
        open(os.path.join(f'{PROJECT_ROOT}/trained_linear_probes_microsoft', model, str(num_layer), 'model.pickle'),
             'rb'))

    batch_files = 1


    for i in range(0, len(test_files), batch_files):

        # Test the linear model on a small subset of activations
        dataset = ActivationsDatasetDynamicPrimaryText(
            test_files[i : i + batch_files],
            root_dir=ROOT_DIR_TEST[model],
            num_layers=(num_layer, num_layer)
        )

        diff = []

        for primary, text in dataset:
            diff.append((text - primary).flatten().float().numpy())

        predict = linear_model.predict(diff)
        poisoned_predicted += np.sum(predict)

    if "clean" in test_files_path:
        print(f"Correct: {total_prompts - poisoned_predicted}  Test accuracy: {(total_prompts - poisoned_predicted) / total_prompts * 100:.2f}%")
    else:
        print(f"Correct: {poisoned_predicted}  Test accuracy: {poisoned_predicted / total_prompts * 100:.2f}%")


def count_microsoft_model_confidence(test_files_path, num_layer):

    """
    Go through all the test activations and create groups based on confidence scores (.5, .6, etc.)
    Let's check how many activation deltas fall in the ranges [.5, .6), [.6, .7) ...

    @param
    test_files_path
    """

    test_files = load_file_paths(test_files_path)

    cnt = defaultdict(int)
    total_instance = 0

    for i in range(len(test_files)):

        dataset = ActivationsDatasetDynamicPrimaryText(
            test_files[i: i + 1],
            root_dir=ROOT_DIR_TEST[model],
            num_layers=(num_layer, num_layer)
        )

        diff = []

        for primary, text in dataset:
            diff.append((text - primary).flatten().float().numpy())

        linear_model = pickle.load(open(os.path.join('../trained_linear_probes_microsoft', model,
                                                     str(num_layer), 'model.pickle'), 'rb'))

        predict_proba = linear_model.predict_proba(diff)
        total_instance += predict_proba.shape[0]

        for x in predict_proba:

            # clean corresponds to x[0]
            # poisoned corresponds to x[1]

            if x[1] >= 0.9:
                cnt[.9] += 1
            elif x[1] >= 0.8:
                cnt[.8] += 1
            elif x[1] >= 0.7:
                cnt[.7] += 1
            elif x[1] >= 0.6:
                cnt[.6] += 1
            elif x[1] >= 0.5:
                cnt[.5] += 1

    print(cnt)


def check_model_consistency(test_files_path):

    """

    This function checks if all the binary classifiers predict the same label for a particular instance.
    Check the similarity of the last four classifiers too, since later layers' activations are more important in task drift detection.

    """

    test_files = load_file_paths(test_files_path)
    same_prediction_across_five_classifiers = 0
    same_prediction_across_last_four_classifiers = 0

    for i in range(len(test_files)):
        prediction_across_layers = []
        for num_layer in LAYERS_PER_MODEL[model]:
            # Test the linear model on a small subset of activations
            dataset = ActivationsDatasetDynamicPrimaryText(
                test_files[i: i + 1],
                root_dir=ROOT_DIR_TEST[model],
                num_layers=(num_layer, num_layer)
            )

            diff = []

            for primary, text in dataset:
                diff.append((text - primary).flatten().float().numpy())

            linear_model = pickle.load(open(os.path.join('../trained_linear_probes_microsoft', model, str(num_layer), 'model.pickle'), 'rb'))
            predict = linear_model.predict(diff)

            prediction_across_layers.append(predict)

        for i in range(len(dataset)):
            if (prediction_across_layers[0][i] == prediction_across_layers[1][i]
                    == prediction_across_layers[2][i] == prediction_across_layers[3][i] == prediction_across_layers[4][i]):
                same_prediction_across_five_classifiers += 1

            if (prediction_across_layers[1][i] == prediction_across_layers[2][i]
                    == prediction_across_layers[3][i] == prediction_across_layers[4][i]):
                same_prediction_across_last_four_classifiers += 1

    print("Same prediction across five classifiers: ", same_prediction_across_five_classifiers)
    print("Same prediction across last four classifiers: ", same_prediction_across_last_four_classifiers)


if __name__ == '__main__':
    # count_microsoft_model_confidence(f'data_files/test_poisoned_files_{model}.txt', 0)

    # check_model_consistency(f'../data_files/test_poisoned_files_{model}.txt')

    clean_filepath = f'{PROJECT_ROOT}/data_files/test_clean_files_{model}.txt'
    poisoned_filepath = f'{PROJECT_ROOT}/data_files/test_poisoned_files_{model}.txt'

    for num_layer in LAYERS_PER_MODEL[model]:

        print(f"Evaluating Layer {num_layer}\n")

        print("On clean dataset")
        print("Microsoft trained linear models:")
        test_microsoft_trained_model(clean_filepath, num_layer)

        print("Adversarially trained linear models:")
        test_adv_trained_model(clean_filepath, num_layer)

        print("-----------------------------------------")

        print("On poisoned dataset")
        print("Microsoft trained linear models:")
        test_microsoft_trained_model(poisoned_filepath, num_layer)

        print("Adversarially trained linear models:")
        test_adv_trained_model(poisoned_filepath, num_layer)

        print("----------------------------------------------------------------------------------------\n\n")

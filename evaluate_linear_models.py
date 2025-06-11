import pickle
from dataset import ActivationsDatasetDynamicPrimaryText
from load_file_paths import load_file_paths
import numpy as np
import os
from collections import defaultdict


np.set_printoptions(suppress=True, linewidth=10000)


def test(test_files_path):
    num_layer = 0
    test_files = load_file_paths(test_files_path)

    # Test the linear model on a small subset of activations
    dataset = ActivationsDatasetDynamicPrimaryText(
        test_files[:2],
        root_dir='/mnt/12EA576EEA574D5B/Activation/phi__3__3.8/test',
        num_layers=(num_layer, num_layer)
    )

    diff = []

    for primary, text in dataset:
        diff.append((text - primary).flatten().float().numpy())

    model = pickle.load(open(os.path.join('./phi3', str(num_layer), 'model.pickle'), 'rb'))
    predict = model.predict(diff)

    poisoned_predicted = np.sum(predict)

    if "clean" in test_files_path:
        print(f"Test accuracy: {(len(dataset) - poisoned_predicted) / len(dataset) * 100:.2f}%")
    else:
        print(f"Test accuracy: {poisoned_predicted / len(dataset) * 100:.2f}%")


def count_microsoft_model_confidence(test_files_path):

    """
    Go through all the test activations and create groups based on confidence scores (.5, .6, etc.)
    Let's check how many activation deltas fall in the ranges [.5, .6), [.6, .7) ...

    @param
    test_files_path
    """

    num_layer = 0
    test_files = load_file_paths(test_files_path)

    cnt = defaultdict(int)
    total_instance = 0

    for i in range(len(test_files)):

        dataset = ActivationsDatasetDynamicPrimaryText(
            test_files[i: i + 1],
            root_dir='/mnt/12EA576EEA574D5B/Activation/llama__3__8B/test',
            num_layers=(num_layer, num_layer)
        )

        diff = []

        for primary, text in dataset:
            diff.append((text - primary).flatten().float().numpy())

        model = pickle.load(open(os.path.join('./trained_linear_probes_microsoft/llama3_8b',
                                              str(num_layer), 'model.pickle'), 'rb'))

        predict_proba = model.predict_proba(diff)
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


if __name__ == '__main__':
    # test('data_files/test_poisoned_files_phi3.txt')
    count_microsoft_model_confidence('data_files/test_poisoned_files_llama3_8b.txt')

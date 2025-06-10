import pickle
from dataset import ActivationsDatasetDynamicPrimaryText
from load_file_paths import load_file_paths
import numpy as np
import os


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


if __name__ == '__main__':
    test('data_files/test_clean_files_phi3.txt')

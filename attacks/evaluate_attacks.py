import matplotlib.pyplot as plt
import json


def plot_epsilon_accuracy(filepath):

    with open(filepath, 'r') as file:
        data = json.load(file)

    for layer in data:
        epsilon = data[layer]['epsilon']
        success_rate = data[layer]['success_rate']

        plt.plot(epsilon, success_rate, label='Layer {}'.format(layer))

    plt.xlabel('epsilon')
    plt.ylabel('Attack success rate')
    plt.legend()
    plt.title(filepath.split('.')[0])

    plt.show()


if __name__ == '__main__':

    attack_type = 'pgd'
    model = 'llama3_8b'

    plot_epsilon_accuracy(f'{attack_type}_attack_details_on_{model}.json')

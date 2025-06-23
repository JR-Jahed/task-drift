import matplotlib.pyplot as plt
import json


def plot_epsilon_accuracy(filepath):

    with open(filepath, 'r') as file:
        data = json.load(file)

    epsilon = data['0']['epsilon']
    x_pos = list(range(len(epsilon)))

    plt.figure(figsize=(10, 6))
    plt.xticks(ticks=x_pos, labels=epsilon)

    for layer in data:
        success_rate = data[layer]['success_rate']

        plt.plot(x_pos, success_rate, label='Layer {}'.format(layer))

    plt.xlabel('epsilon')
    plt.ylabel('Attack success rate')
    plt.title(filepath.split('.')[0])
    plt.legend(loc='center left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    attack_type = 'pgd'
    model = 'llama3_8b'

    plot_epsilon_accuracy(f'{attack_type}_attack_details_on_{model}.json')

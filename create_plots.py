import matplotlib.pyplot as plt


def plot(data, plot_title, xlabel, ylabel):
    x = [int(k) for k in data.keys()]
    y = data.values()

    fig, ax = plt.subplots()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel, labelpad=15)

    bars = ax.bar(x, y, width=.6, align='center', edgecolor='black')

    ax.set_xticks(x)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom'
        )

    fig.suptitle(plot_title, y=.96)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    plt.show()



total_number_of_prompts_misclassified_by_a_specific_number_of_classifiers_without_suffix = {'0': 17612, '1': 13025, '2': 413, '3': 65, '4': 15, '5': 4}
total_number_of_prompts_misclassified_by_a_specific_number_of_classifiers_with_suffix = {'0': 0, '1': 0, '2': 4, '3': 42, '4': 1650, '5': 29438}

plot(
    total_number_of_prompts_misclassified_by_a_specific_number_of_classifiers_without_suffix,
    'Total number of prompts misclassified by a specific\n number of classifiers without suffix (Phi)',
    'Total number of classifiers',
    'Total number of prompts that were misclassified'
)

plot(
    total_number_of_prompts_misclassified_by_a_specific_number_of_classifiers_with_suffix,
    'Total number of prompts misclassified by a specific\n number of classifiers with suffix (Phi)',
    'Total number of classifiers',
    'Total number of prompts that were misclassified'
)


layerwise_misclassification_without_suffix = {'0': 13273, '7': 456, '15': 172, '23': 25, '31': 200}
layerwise_misclassification_with_suffix = {'0': 29735, '7': 31115, '15': 31128, '23': 30815, '31': 31131}

plot(
    layerwise_misclassification_without_suffix,
    'Layer wise misclassification without suffix (Phi)',
    'Number of layer',
    'Total misclassification by that classifier'
)

plot(
    layerwise_misclassification_with_suffix,
    'Layer wise misclassification with suffix (Phi)',
    'Number of layer',
    'Total misclassification by that classifier'
)
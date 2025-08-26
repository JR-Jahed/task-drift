import json
from constants import PROJECT_ROOT


def analyse_adv_trained_models(filepath):

    """

    Result list: [
        {
            Suffix: string,
            Prompt indices: [],
            Attack result list: [
                {
                    Without suffix: {
                        labels: []
                        probs: [[]]
                    },
                    With suffix: {
                        labels: []
                        probs: [[]]
                    }
                }
            ],
            Total number of prompts correctly classified by a specific number of classifiers: {
                Without suffix: {
                    0: int, 1: int, 2: int, 3: int, 4: int, 5: int
                },
                With suffix: {
                    0: int, 1: int, 2: int, 3: int, 4: int, 5: int
                }
            },
            Layerwise correct classification: {
                Without suffix: {
                    0: int, 7: int, 15: int, 23: int, 31: int
                },
                With suffix: {
                    0: int, 7: int, 15: int, 23: int, 31: int
                }
            }
        }
    ]

    """

    data = json.load(open(filepath, 'r'))

    print(f"Total suffixes: {len(data['Result list'])}")
    print(f"Tested on: {len(data['Result list'][0]['Prompt indices'])} prompts\n")

    num_test_prompts = len(data['Result list'][0]['Prompt indices'])

    for i in range(len(data['Result list'])):
        print(data['Result list'][i]['Suffix'])
        print(data['Result list'][i]['Total number of prompts correctly classified by a specific number of classifiers']['Without suffix'])
        print(data['Result list'][i]['Total number of prompts correctly classified by a specific number of classifiers']['With suffix'])

        print(data['Result list'][i]['Layerwise correct classification']['Without suffix'])
        print(data['Result list'][i]['Layerwise correct classification']['With suffix'])

        print("\nWithout suffix  ", end='')

        for k, v in data['Result list'][i]['Layerwise correct classification']['Without suffix'].items():
            print(f'----  {k}: {(v / num_test_prompts) * 100:6.2f}%  ', end='')

        print("\nWith suffix     ", end='')

        for k, v in data['Result list'][i]['Layerwise correct classification']['With suffix'].items():
            print(f'----  {k}: {(v / num_test_prompts) * 100:6.2f}%  ', end='')

        print('\n\n')


def analyse_optimisation(filepath):

    """

    Result list: [
        {
            Initial suffix: string,
            Prompt indices: [],
            Result: [
                {
                    Iteration log: [
                        {
                            losses: [[]],
                            labels: [[]],
                            probs: [[]],
                            suffix: string
                        }
                    ],
                    Total time: float,
                    No suffix found after filtering: Boolean     (only for LLaMA)
                }
            ]
        }
    ]

    """

    def format_probs(probs):
        formatted_probs = []
        for prob_pair in probs:
            formatted_pair = [f"{p:.8f}" for p in prob_pair]
            formatted_probs.append(f"[{formatted_pair[0]}, {formatted_pair[1]}]")
        probs_str = "[" + ", ".join(formatted_probs) + "]"

        return probs_str

    def format_losses(losses):
        losses_str = "[" + ", ".join([f'{loss:.8f}' for loss in losses]) + "]"
        return losses_str

    data = json.load(open(filepath, 'r'))
    print(f"Total suffixes: {len(data['Result list'])}")

    suffix_index = 6

    print(f"Current suffix index: {suffix_index}\n")

    for i, iteration_log in enumerate(data['Result list'][suffix_index]['Result']['Iteration log']):

        print(f"i: {i}")

        for loss, labels, probs in zip(iteration_log['losses'], iteration_log['labels'], iteration_log['probs']):
            print(f"losses: {format_losses(loss)} labels: {labels} probs: {format_probs(probs)}")

        print(iteration_log['suffix'])
        print("-------------------------------------------------------\n")




if __name__ == "__main__":

    # filepath = f'{PROJECT_ROOT}/test_results/phi3_result_adv_trained_models.json'
    # # filepath = '/home/40456997@eeecs.qub.ac.uk/Test Results/Adv Trained Models/Epoch 3/phi3_result_adv_trained_models_on_initial_test_suffixes.json'
    #
    # analyse_adv_trained_models(filepath)

    filepath = f'{PROJECT_ROOT}/opt_results/phi3_optimisation_result.json'

    analyse_optimisation(filepath)




# total_prompts = len(data['Attack result list'])
# same_prediction_across_last_four_classifiers = 0
#
# for i in range(len(data['Attack result list'])):
#     labels = data['Attack result list'][i]['With suffix']['labels']
#
#     if labels[1] == 0 and labels[2] == 0 and labels[3] == 0 and labels[4] == 0:
#         same_prediction_across_last_four_classifiers += 1
#
# print(f"Same prediction across last four classifiers: {same_prediction_across_last_four_classifiers}"
#       f"     ASR: {round(same_prediction_across_last_four_classifiers / total_prompts * 100, 2)}%")
#
#
# asr_all_five = data["Total number of prompts misclassified by a specific number of classifiers"]["With suffix"]['5']
# asr_four_or_more = data["Total number of prompts misclassified by a specific number of classifiers"]["With suffix"]['4'] + asr_all_five
# asr_three_or_more = data["Total number of prompts misclassified by a specific number of classifiers"]["With suffix"]['3'] + asr_four_or_more
#
# print(f"\nASR all five: {round(asr_all_five / total_prompts * 100, 2)}%")
# print(f"ASR four or more: {round(asr_four_or_more / total_prompts * 100, 2)}%")
# print(f"ASR three or more: {round(asr_three_or_more / total_prompts * 100, 2)}%")

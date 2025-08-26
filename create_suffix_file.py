import json
from constants import PROJECT_ROOT
import torch


iteration_log_index = [
    [-1, 27],
    [-1, 62],
    [-1, 70],
    [-1, 41],
    [-1, 41],
    [-1, 33],
    [],
    [-1, 65],
    [-1, 67],
    [-1, 74],
    [-1, 84]
]

filepath = f'{PROJECT_ROOT}/opt_results/phi3_optimisation_result.json'

data = json.load(open(filepath, 'r'))

total_suffixes = len(data['Result list'])

suffix_list = []

for i, suffix_indices in enumerate(iteration_log_index):
    if len(suffix_indices) != 0:
        suffix_list.append(data['Result list'][i]['Result']['Iteration log'][suffix_indices[0]]['suffix'])
        suffix_list.append(data['Result list'][i]['Result']['Iteration log'][suffix_indices[1]]['suffix'])

total_prompts = 50

a = torch.arange(0, len(suffix_list), len(suffix_list) / total_prompts).type(torch.int)

print(a)
print(a.shape)

suffix_list_json = {
    "Suffix list": suffix_list,
}

suffix_list_filepath = f'{PROJECT_ROOT}/generate_activations/data/suffix_list.json'
with open(suffix_list_filepath, 'w') as f:
    json.dump(suffix_list_json, f)
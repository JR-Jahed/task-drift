import json
from constants import PROJECT_ROOT


model = 'phi3'


iteration_log_index = {
    'train': {
        'suffix indices': [
            0, 1, 2, 3, 4, 5, 7, 8, 9, 10
        ],
        'iteration_list': [
            [-1, 27],
            [-1, 62],
            [-1, 70],
            [-1, 41],
            [-1, 41],
            [-1, 33],
            [-1, 65],
            [-1, 67],
            [-1, 74],
            [-1, 84],
        ]
    },
    'val': {
        'suffix indices': [
            11, 12
        ],
        'iteration_list': [
            [-1, 16],
            [-1, 21],
        ]
    },
    'test': {
        'suffix indices': [
            13, 14, 15, 16, 17, 18,
        ],
        'iteration_list': [
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
        ]
    }
}

filepath = f'{PROJECT_ROOT}/opt_results/{model}_optimisation_result.json'

data = json.load(open(filepath, 'r'))

total_suffixes = len(data['Result list'])

suffix_list = {
    'train': [],
    'val': [],
    'test': []
}

for suffix_type, value in iteration_log_index.items():
    for suffix_index, iteration_indices in zip(value['suffix indices'], value['iteration_list']):

        for iteration_index in iteration_indices:
            suffix_list[suffix_type].append(data['Result list'][suffix_index]['Result']['Iteration log'][iteration_index]['suffix'])


train_suffix_list_json = {
    "Suffix list": suffix_list['train'],
}

val_suffix_list_json = {
    "Suffix list": suffix_list['val'],
}

test_suffix_list_json = {
    "Suffix list": suffix_list['test'],
}

print(train_suffix_list_json)
print(val_suffix_list_json)
print(test_suffix_list_json)


train_suffix_list_filepath = f'{PROJECT_ROOT}/generate_activations/data/{model}_train_suffix_list.json'
val_suffix_list_filepath = f'{PROJECT_ROOT}/generate_activations/data/{model}_val_suffix_list.json'
test_suffix_list_filepath = f'{PROJECT_ROOT}/{model}_test_suffix_list.json'

# with open(train_suffix_list_filepath, 'w') as f:
#     json.dump(train_suffix_list_json, f, indent=4)

# with open(val_suffix_list_filepath, 'w') as f:
#     json.dump(val_suffix_list_json, f, indent=4)

# with open(test_suffix_list_filepath, 'w') as f:
#     json.dump(test_suffix_list_json, f, indent=4)

import json

"""

The json file has been saved as

file[num layer][epsilon][filepath] = list of indices

"""


filepath = '../attacks/pgd_indices_of_successful_attacks_on_phi3.json'

file = json.load(open(filepath, 'r'))

indices = file['31']['epsilon_0.02']['poisoned_hidden_states_0_1000_20240717_115335.pt']

print(indices)

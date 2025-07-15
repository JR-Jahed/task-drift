import pickle
import numpy as np
import torch

np.set_printoptions(suppress=True)

"""
task drift = 1
no task drift = 0
"""


model = pickle.load(
    open('../trained_linear_probes_microsoft/phi3/31/model.pickle', 'rb')
)

filepath = '/mnt/12EA576EEA574D5B/Activation/phi__3__3.8/test/poisoned_hidden_states_0_1000_20240717_115335.pt'
activations = torch.load(filepath)

primary_activation = activations[0][25][-1]

poisoned_activation = torch.tensor([], dtype=torch.bfloat16)
# Target activation              tensor([], dtype=torch.bfloat16)
adv_activation =      torch.tensor([], dtype=torch.bfloat16)


delta = (adv_activation - primary_activation).to(torch.float32).detach().numpy().reshape(1, -1)

label = model.predict(delta)
prob = model.predict_proba(delta)

print(label)
print(prob)

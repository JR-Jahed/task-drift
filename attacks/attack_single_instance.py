import pickle
import os
from attacks.attack import get_correctly_classified_activations
from attacks.pgd import pgd
import torch
import io
import contextlib

torch.set_printoptions(sci_mode=False, linewidth=100000, threshold=float('inf'))


model = 'phi3'


def format_and_print_tensor(tensor):
    # Redirect stdout to capture print output
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        print(tensor)
        tensor_str = buf.getvalue()

    tensor_str = tensor_str.replace(' ', '').replace(',', ', ')
    print(tensor_str)


def attack_single_instance(num_layer, index, epsilon):

    linear_model = pickle.load(open(os.path.join('../trained_linear_probes_microsoft',
                                                 model, str(num_layer), 'model.pickle'), 'rb'))

    activations, indices = get_correctly_classified_activations(["poisoned_hidden_states_0_1000_20240717_115335.pt"],
                                           num_layer, linear_model)

    success, target_activation = pgd(linear_model, activations[index], 0, epsilon=epsilon, alpha=.01, num_iter=40)

    print("\n\nPoisoned Activation")
    format_and_print_tensor(activations[index][1])

    print("Target activation")
    format_and_print_tensor(target_activation)

    print(f"MSE loss: {torch.nn.functional.mse_loss(activations[index][1].to(torch.float32), target_activation.to(torch.float32)):.10f}")
    print(f"Cosine similarity: {1 - torch.nn.functional.cosine_similarity(activations[index][1].to(torch.float32),target_activation.to(torch.float32), dim=0):.10f}")


if __name__ == "__main__":
    attack_single_instance(num_layer=31, index=25, epsilon=.02)

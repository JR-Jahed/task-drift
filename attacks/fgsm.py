import numpy as np
import torch
import torch.nn.functional as F

np.set_printoptions(suppress=True, linewidth=10000)


def fgsm(model, activations, label, epsilon=.01):

    print("------------------------------------------------------------------")
    print("FGSM\n")

    delta = (activations[1] - activations[0]).detach().clone().requires_grad_(True)

    weights_np = model.coef_[0]
    bias_np = model.intercept_[0]

    weights = torch.tensor(weights_np, dtype=torch.float32)
    bias = torch.tensor(bias_np, dtype=torch.float32)

    z = torch.dot(weights, delta) + bias
    pred = torch.sigmoid(z)

    label_tensor = torch.tensor(label, dtype=torch.float32)
    loss = F.binary_cross_entropy(pred, label_tensor)

    loss.backward()
    grad = delta.grad

    delta_adv = delta + epsilon * torch.sign(grad)
    delta_adv = delta_adv.detach()

    delta_np = delta.detach().numpy()
    delta_adv_np = delta_adv.numpy()

    original_probability = model.predict_proba([delta_np])[0]
    adversarial_probability = model.predict_proba([delta_adv_np])[0]

    print("original prob   ", original_probability)
    print("adv prob        ", adversarial_probability)
    print("\nadv success   ", adversarial_probability[1 - label] > adversarial_probability[label])

    print("------------------------------------------------------------------")

    return adversarial_probability[1 - label] > adversarial_probability[label]
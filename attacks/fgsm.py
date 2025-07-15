import numpy as np
import torch

np.set_printoptions(suppress=True, linewidth=10000)


def fgsm(model, activations, target_label, epsilon=.01):

    print("------------------------------------------------------------------")
    print("FGSM\n")

    delta = (activations[1] - activations[0]).detach().clone().to(torch.float32).requires_grad_(True)

    weights_np = model.coef_[0]
    bias_np = model.intercept_[0]

    weights = torch.tensor(weights_np, dtype=torch.float32)
    bias = torch.tensor(bias_np, dtype=torch.float32)

    logits = torch.dot(weights, delta) + bias

    target_label_tensor = torch.tensor(target_label, dtype=torch.float32)
    loss = torch.nn.BCEWithLogitsLoss()(logits, target_label_tensor)

    loss.backward()
    grad = delta.grad

    delta_adv = delta - epsilon * torch.sign(grad)
    delta_adv = delta_adv.detach()

    delta_np = delta.detach().numpy()
    delta_adv_np = delta_adv.numpy()

    original_probability = model.predict_proba([delta_np])[0]
    adversarial_probability = model.predict_proba([delta_adv_np])[0]

    print("original prob   ", original_probability)
    print("adv prob        ", adversarial_probability)
    print("\nadv success   ", adversarial_probability[target_label] > .5)

    print("------------------------------------------------------------------")

    target_activation = delta_adv.to(activations[0].dtype) + activations[0]

    return adversarial_probability[target_label] > .5, target_activation
import numpy as np
import torch
import torch.nn.functional as F

np.set_printoptions(suppress=True, linewidth=10000)


def pgd(model, activations, label, epsilon=.1, alpha=.01, num_iter=20):

    print("------------------------------------------------------------------")
    print("PGD\n")

    delta_original = (activations[1] - activations[0]).detach().clone()
    delta_adv = delta_original.clone().detach().requires_grad_(True)

    weights_np = model.coef_[0]
    bias_np = model.intercept_[0]

    weights = torch.tensor(weights_np, dtype=torch.float32)
    bias = torch.tensor(bias_np, dtype=torch.float32)

    label_tensor = torch.tensor(label, dtype=torch.float32)

    for i in range(num_iter):
        z = torch.dot(weights, delta_adv) + bias
        prediction = torch.sigmoid(z)

        loss = F.binary_cross_entropy(prediction, label_tensor)

        if delta_adv.grad is not None:
            delta_adv.grad.zero_()

        loss.backward()
        grad = delta_adv.grad

        delta_adv = delta_adv + alpha * torch.sign(grad)

        delta_adv = torch.max(torch.min(delta_adv, delta_original + epsilon), delta_original - epsilon)

        delta_adv = delta_adv.detach().clone().requires_grad_(True)

    # Evaluate with scikit-learn
    delta_np = delta_original.detach().numpy()
    delta_adv_np = delta_adv.detach().numpy()

    original_probability = model.predict([delta_np])[0]
    adversarial_probability = model.predict_proba([delta_adv_np])[0]

    print("original prob   ", original_probability)
    print("adv prob        ", adversarial_probability)
    print("\nadv success   ", adversarial_probability[1 - label] > .5)
    print("------------------------------------------------------------------")

    return adversarial_probability[1 - label] > .5

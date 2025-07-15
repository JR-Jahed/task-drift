import numpy as np
import torch

np.set_printoptions(suppress=True, linewidth=10000)


def pgd(model, activations, target_label=0, epsilon=.1, alpha=.01, num_iter=20):

    """

    Perform a targeted PGD attack on the linear model

    :param model: The Microsoft trained logistic regression model
    :param activations: The test activation pair (primary, text)
    :param target_label: The targeted class of the attack
    :param epsilon:
    :param alpha:
    :param num_iter:

    :return:
    attack_success: A boolean value that represents whether the attack was successful
    target_activation: The target activation after the attack
    """


    print("------------------------------------------------------------------")
    print("PGD\n")

    delta_original = (activations[1] - activations[0]).detach().clone().to(torch.float32)
    delta_adv = delta_original.clone().detach().requires_grad_(True)

    weights_np = model.coef_[0]
    bias_np = model.intercept_[0]

    weights = torch.tensor(weights_np, dtype=torch.float32)
    bias = torch.tensor(bias_np, dtype=torch.float32)

    criterion = torch.nn.BCEWithLogitsLoss()

    target_label_tensor = torch.tensor(target_label, dtype=torch.float32)

    for i in range(num_iter):
        logits = torch.dot(weights, delta_adv) + bias

        if delta_adv.grad is not None:
            delta_adv.grad.zero_()

        loss = criterion(logits, target_label_tensor)

        loss.backward()
        grad = delta_adv.grad

        delta_adv = delta_adv - alpha * torch.sign(grad)

        delta_adv = torch.max(torch.min(delta_adv, delta_original + epsilon), delta_original - epsilon)

        delta_adv = delta_adv.detach().clone().requires_grad_(True)

    # Evaluate with scikit-learn
    delta_np = delta_original.detach().numpy()
    delta_adv_np = delta_adv.detach().numpy()

    original_probability = model.predict_proba([delta_np])[0]
    adversarial_probability = model.predict_proba([delta_adv_np])[0]

    print("original prob   ", original_probability)
    print("adv prob        ", adversarial_probability)
    print("\nadv success   ", adversarial_probability[target_label] > .5)
    print("------------------------------------------------------------------")

    target_activation = delta_adv.detach().to(activations[0].dtype) + activations[0]

    return adversarial_probability[target_label] > .5, target_activation

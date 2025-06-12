import numpy as np

np.set_printoptions(suppress=True, linewidth=10000)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def pgd(model, activations, label, epsilon=.1, alpha=.01, num_iter=20):

    print("------------------------------------------------------------------")
    print("PGD\n")

    activations_ = activations.copy()

    delta_original = activations_[1] - activations_[0]
    delta_adv = delta_original.copy()

    original_probability = model.predict_proba([delta_adv])[0]
    adversarial_probability = None

    weights = model.coef_[0]
    bias = model.intercept_[0]

    for i in range(num_iter):

        z = np.dot(weights, delta_adv) + bias
        prediction = sigmoid(z)

        gradient_delta_adv = (prediction - label) * weights
        delta_adv = delta_adv + alpha * np.sign(gradient_delta_adv)

        delta_adv = np.clip(delta_adv, delta_original - epsilon, delta_original + epsilon)

        adversarial_probability = model.predict_proba([delta_adv])[0]

        if adversarial_probability[1 - label] > .5:
            print("Alpha: ", (i + 1) * alpha)
            break

    print("original prob   ", original_probability)
    print("adv prob        ", adversarial_probability)
    print("\nadv success   ", adversarial_probability[1 - label] > .5)
    print("------------------------------------------------------------------")

    return adversarial_probability[1 - label] > .5

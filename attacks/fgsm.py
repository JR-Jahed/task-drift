import numpy as np

np.set_printoptions(suppress=True, linewidth=10000)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fgsm(model, activations, label, epsilon=.01):

    print("------------------------------------------------------------------")
    print("FGSM\n")

    delta = activations[1] - activations[0]

    weights = model.coef_[0]
    bias = model.intercept_[0]

    z = np.dot(weights, delta) + bias
    prediction = sigmoid(z)

    gradient_delta = (prediction - label) * weights
    delta_adv = delta + epsilon * np.sign(gradient_delta)

    original_probability = model.predict_proba([delta])[0]
    adversarial_probability = model.predict_proba([delta_adv])[0]

    print("original prob   ", original_probability)
    print("adv prob        ", adversarial_probability)
    print("\nadv success   ", adversarial_probability[1 - label] > adversarial_probability[label])

    print("------------------------------------------------------------------")

    return adversarial_probability[1 - label] > adversarial_probability[label]
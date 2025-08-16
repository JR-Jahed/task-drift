import torch
import torch.nn as nn


@torch.no_grad()
def clamp_linf(x, x0, eps):
    # Project x back to L-inf ball around x0 of radius eps
    return torch.max(torch.min(x, x0 + eps), x0 - eps)


def pgd_torch_linear(model, primary, poisoned, epsilon=0.1, alpha=0.01, steps=20, target_label=0):
    """
    Targeted PGD that perturbs the poisoned activation so the linear model predicts target_label (default: 0, 'clean').
    Assumes:
      - primary, poisoned: 1D tensors of shape [D] on same device/dtype as model
      - model(x) returns logits of shape [1] for binary classification
    Returns:
      adv_poisoned (1D tensor [D])
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    primary = primary.to(device=device, dtype=dtype)
    x0 = poisoned.to(device=device, dtype=dtype)  # start from the original poisoned activation
    x = x0.clone().detach().requires_grad_(True)

    y_t = torch.tensor(float(target_label), device=device, dtype=dtype)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(steps):
        # loss on delta = (x - primary)
        delta = (x - primary)  # shape [D]
        logits = model(delta.unsqueeze(0)).squeeze(0)  # [1] -> scalar
        loss = criterion(logits, y_t)

        if x.grad is not None:
            x.grad.zero_()
        loss.backward()

        # targeted toward label 0 => minimize loss wrt target 0 => descend
        with torch.no_grad():
            x = x - alpha * torch.sign(x.grad)
            x = clamp_linf(x, x0, epsilon)
        x.requires_grad_(True)

    return x.detach()  # adv_poisoned (1D)
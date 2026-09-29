# Gradients and Hessians
import torch


def gradient(f, x):
    return torch.autograd.functional.jacobian(
        f,
        x,
        vectorize=False,
    )

def hessian(f, x):
    return torch.autograd.functional.hessian(
        f,
        x,
        vectorize=False,
    )
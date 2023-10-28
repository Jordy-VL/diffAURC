from collections import OrderedDict
import numpy as np
from scipy.special import softmax
import torch

##### g: confidence scoring functions #####


def entropy(x):
    exp_x = np.exp(x)
    A = np.sum(exp_x, axis=-1)  # sum of exp(x_i)
    B = np.sum(x * exp_x, axis=-1)  # sum of x_i * exp(x_i)
    return np.log(A) - B / A


def entropy_torch(x):
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=-1)  # sum of exp(x_i)
    B = torch.sum(x * exp_x, dim=-1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B / A


def top12_margin(x):
    """top-1 - top-2"""
    values, _ = torch.topk(x, k=2, dim=-1)
    if x.ndim == 1:
        return values[0] - values[1]
    return values[:, 0] - values[:, 1]


def top12_margin_np(x):
    values = np.sort(x, axis=-1)
    if x.ndim == 1:
        return values[0] - values[1]
    return values[:, 0] - values[:, 1]


CSF_dict = OrderedDict(
    {
        "msp": lambda x: torch.max(torch.nn.functional.softmax(x, dim=-1), -1)[0],  # maximum softmax probability
        "entropy": lambda x: -entropy_torch(x),  # negative entropy
        "margin": lambda x: top12_margin(x),  # top-1 - top-2
        "msp_np": lambda x: np.max(softmax(x, axis=-1), -1),  # maximum softmax probability
        "entropy_np": lambda x: -entropy(x),  # negative entropy
        "margin_np": lambda x: top12_margin_np(x),  # top-1 - top-2
    }
)


##### differentiable loss #####


def zero_one_loss(input, target):
    """0-1 loss (inverse)"""
    return (torch.argmax(input, dim=-1) != target).float().mean()


class AURCLoss(torch.nn.Module):

    """PyTorch version of differentiable AURC (https://arxiv.org/pdf/1805.08206.pdf)

    \operatorname{AURC}(f) := \frac{1}{n} \sum_{i=1}^{n-1} \frac{\sum_{j=i+1}^n\ell([f(x_{i:n})]_{1:k},y_j)}{(n-i)}

    This version has time complexity O(N), yet does not take into account duplicate values in g(f_X)
    """

    def __init__(self, g=CSF_dict["msp"], loss_function=torch.nn.CrossEntropyLoss()):
        super(AURCLoss, self).__init__()
        self.g = g
        self.loss_function = loss_function

    def forward(self, input, target):
        B = len(target)
        indices_sorted = torch.argsort(self.g(input), descending=True)
        sorted_f_X = input[indices_sorted]
        sorted_y = target[indices_sorted]
        final_sum = 0
        partial_loss = self.loss_function(sorted_f_X[0], sorted_y[0])  # initialize by largest (often 0 anyway)
        for i in range(1, B):
            final_sum += partial_loss / i
            partial_loss += self.loss_function(sorted_f_X[i], sorted_y[i])
        return final_sum / B


class AlphaAURCLoss(torch.nn.Module):

    """PyTorch version of differentiable AURC (https://arxiv.org/pdf/1805.08206.pdf)

    \operatorname{AURC}(f) := \frac{1}{n} \sum_{i=1}^{n-1} \alpha_i \ell([f(x_{i:n})]_{1:k},y_j)}

    This version has time complexity O(N), yet does not take into account duplicate values in g(f_X)
    """

    def __init__(self, g=CSF_dict["msp"], loss_function=torch.nn.CrossEntropyLoss()):
        super(AlphaAURCLoss, self).__init__()
        self.g = g
        self.loss_function = loss_function

    def forward(self, input, target):
        N = len(target)
        indices_sorted = torch.argsort(self.g(input), descending=False)
        losses = self.loss_function(input[indices_sorted], target[indices_sorted])
        alphas = torch.Tensor([-np.log(1 - i / N) for i in range(N)])
        return torch.mean(alphas * losses)
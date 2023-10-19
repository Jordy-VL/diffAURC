from collections import OrderedDict
import numpy as np
from scipy.special import softmax
import pickle
from sklearn.model_selection import train_test_split
import torch

from AURC_implementations import IMPLEMENTATIONS, runtime_wrapper

##### data generation for testing #####


def generate_random_data():
    num_samples = 10
    p_test = np.random.rand(num_samples, 3)
    y_test = np.random.choice(3, num_samples)
    return p_test, y_test


def generate_random_data_with_confidence(N=10, K=3, concentration=1, onehot=False):
    # N evaluation instances {(x_i,y_i)}_{i=1}^N

    def random_mc_instance(concentration=1, onehot=False):
        reference = np.argmax(np.random.dirichlet(([concentration for _ in range(K)])), -1)  # class targets
        prediction = np.random.dirichlet(([concentration for _ in range(K)]))  # probabilities
        if onehot:
            reference = np.eye(K)[np.argmax(reference, -1)]
        return reference, prediction

    y_test, p_test = list(zip(*[random_mc_instance() for i in range(N)]))
    y_test = np.array(y_test, dtype=np.int64)
    p_test = np.array(p_test, dtype=np.float32)
    return p_test, y_test


def load_CIFAR_logits(path="../data/resnet110_c10_logits.p"):
    """
    This testing script loads actual probabilisitic predictions from a resnet finetuned on CIFAR

    There are a number of logits-groundtruth pickles available @ https://github.com/markus93/torch.NN_calibration/tree/master/logits
    [Seems to have moved from Git-LFS to sharepoint]
    https://tartuulikool-my.sharepoint.com/:f:/g/personal/markus93_ut_ee/EmW0xbhcic5Ou0lRbTrySOUBF2ccSsN7lo6lvSfuG1djew?e=l0TErb

    See https://github.com/markus93/NN_calibration/blob/master/logits/Readme.txt to decode the [model_dataset] filenames
    """

    def unpickle_probs(file, verbose=0, normalize=False):
        with open(file, "rb") as f:  # Python 3: open(..., 'rb')
            y1, y2 = pickle.load(f)  # unpickle the content

        if isinstance(y1, tuple):
            y_probs_val, y_val = y1
            y_probs_test, y_test = y2
        else:
            y_probs_val, y_probs_test, y_val, y_test = train_test_split(
                y1, y2.reshape(-1, 1), test_size=len(y2) - 5000, random_state=15
            )  # Splits the data in the ca%load_ext autoreload

        if normalize:
            y_probs_val = softmax(y_probs_val, -1)
            y_probs_test = softmax(y_probs_test, -1)

        if verbose:
            print("y_probs_val:", y_probs_val.shape)  # (5000, 10); Validation set probabilities of predictions
            print("y_true_val:", y_val.shape)  # (5000, 1); Validation set true labels
            print("y_probs_test:", y_probs_test.shape)  # (10000, 10); Test set probabilities
            print("y_true_test:", y_test.shape)  # (10000, 1); Test set true labels

        return ((y_probs_val, y_val.ravel()), (y_probs_test, y_test.ravel()))

    (p_val, y_val), (p_test, y_test) = unpickle_probs(path, verbose=0)
    return (p_val, y_val), (p_test, y_test)


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
        partial_loss = self.loss_function(sorted_f_X[0], sorted_y[0])
        for i in range(1, B):
            final_sum += partial_loss / i
            partial_loss += self.loss_function(sorted_f_X[i], sorted_y[i])
        return final_sum / B


if __name__ == "__main__":
    # add argparse for N and K
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--real", action="store_true")  # sample from CIFAR (with replacement)
    args = parser.parse_args()

    if args.real:
        (p_test, y_test), _ = load_CIFAR_logits()
        args.K = 10
        if args.N != len(y_test):  # 10000
            sub_n = np.random.choice(list(range(len(y_test))), args.N)  # replace =True if larger
            p_test, y_test = p_test[sub_n], y_test[sub_n]
        else:
            args.N = len(y_test)
    else:
        p_test, y_test = generate_random_data_with_confidence(N=args.N, K=args.K, concentration=1, onehot=False)

    #baseline accuracy
    print(f"baseline accuracy: {np.mean(p_test.argmax(-1) == y_test)}")
    
    p_test = torch.from_numpy(p_test).requires_grad_()
    y_test = torch.from_numpy(y_test)

    # all options: g, loss function, metric implementations
    loss_function = torch.nn.CrossEntropyLoss()
    for CSF, g in CSF_dict.items():
        if "_np" in CSF:
            continue
        loss_fx = AURCLoss(g=g, loss_function=loss_function)
        loss = loss_fx(p_test, y_test)
        loss.backward()
        # print(p_test.grad)

        # test metrics vs. loss function
        for _, impl in IMPLEMENTATIONS:
            g = CSF_dict[CSF + "_np"]
            v, t = runtime_wrapper(impl, p_test.detach().numpy(), g, y_test.numpy())
            print(f"{CSF}: loss: {loss.item()} vs. {impl.__name__}: {v} taking {round(t,4)}s")

from collections import OrderedDict
import numpy as np
from scipy.special import softmax
import pickle
from sklearn.model_selection import train_test_split
import torch

from AURC_loss import AURCLoss, CSF_dict
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


def test_all_options(p_test, y_test):
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
        for key, impl in IMPLEMENTATIONS:
            if args.N > 1000 and key == "naive":
                continue
            g = CSF_dict[CSF + "_np"]
            v, t = runtime_wrapper(impl, p_test.detach().numpy(), g, y_test.numpy())
            print(f"{CSF}: loss: {loss.item()} vs. {impl.__name__}: {v} taking {round(t,4)}s")


def test_consistency(p_test, y_test):
    def batch(iterable, n=1):
        length = len(iterable)
        for ndx in range(0, length, n):
            yield iterable[ndx : min(ndx + n, length)]

    #np.random.seed(args.seed)
    
    # random batching with different N; now extend to different seeds
    collection = OrderedDict()
    std_errors = OrderedDict()
    NN = [2, 5, 10, 20, 50, 100, 1000, 2500, 5000, 10000]
    for Ns in NN:
        collection[Ns] = []
        std_errors[Ns] = []
        for seed in NN + list(range(20000, 20200)):
            permutation = np.random.RandomState(seed=seed).permutation(len(y_test))
            p_test_bs, y_test_bs = p_test[permutation], y_test[permutation]
            p_test_bs, y_test_bs = batch(p_test, Ns), batch(y_test, Ns)

            batched_losses = [AURCLoss()(p_test_batch, y_test_batch).detach().numpy() for p_test_batch, y_test_batch in zip(p_test_bs, y_test_bs)]
            batched_loss = np.mean(batched_losses)
            collection[Ns].append(batched_loss)
            std_errors[Ns].append(np.std(batched_losses) / np.sqrt(len(batched_losses)))
            
        collection[Ns] = np.mean(collection[Ns])
        std_errors[Ns] = np.mean(std_errors[Ns]) #average std error over seeds
        
        print(f"Batched: N={Ns}, loss: {collection[Ns]}, std: {std_errors[Ns]}")
        
        
    # full
    loss_fx = AURCLoss()
    loss = loss_fx(p_test, y_test)
    collection[20000] = loss.item()
    std_errors[20000] = 0
    print(f"Population: loss: {loss.item()}")
    
    #plot collection over Ns (with std errors) in plotly
    
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(collection.keys()), y=list(collection.values()), name="Batched",  error_y=dict(type='data', array=list(std_errors.values()), visible=True)))
    fig.add_trace(go.Scatter(x=list(collection.keys()), y=[collection[20000]]*len(collection), mode='lines', line=dict(dash='dash'), name="Population"))
    fig.update_layout(xaxis_title="N", yaxis_title="AURC")
    fig.update_layout(xaxis_type="log")
    fig.update_layout(legend_title_text="Ablation", legend_title_font_color="green")
    
    fig.show()
    
    
    # import matplotlib.pyplot as plt
    # plt.errorbar(list(collection.keys()), list(collection.values()), yerr=list(std_errors.values()), fmt='o')
    # plt.plot(list(collection.keys()), [collection[20000]]*len(collection), linestyle='--')
    # plt.xlabel("N")
    # plt.ylabel("AURC")
    # plt.show()

    
    

if __name__ == "__main__":
    # add argparse for N and K
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--real", action="store_true")  # sample from CIFAR (with replacement)
    parser.add_argument("--seed", type=int, default=42)  # sample from CIFAR (with replacement)
    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.real:
        (p_test, y_test), (p_val, y_val) = load_CIFAR_logits()
        y_test = np.hstack((y_test, y_val))
        p_test = np.vstack((p_test, p_val))
        args.K = 10
        if args.N != len(y_test):  # 20000
            sub_n = np.random.choice(list(range(len(y_test))), args.N)  # replace =True if larger
            p_test, y_test = p_test[sub_n], y_test[sub_n]
        else:
            args.N = len(y_test)
    else:
        p_test, y_test = generate_random_data_with_confidence(N=args.N, K=args.K, concentration=1, onehot=False)

    # baseline accuracy
    print(f"baseline accuracy: {np.mean(p_test.argmax(-1) == y_test)}")

    p_test = torch.from_numpy(p_test).requires_grad_()
    y_test = torch.from_numpy(y_test)
    
    test_consistency(p_test, y_test)
    
    test_all_options(p_test, y_test)

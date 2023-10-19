import numpy as np
from argparse import Namespace
import math
import time
import pandas as pd


def runtime_wrapper(input_function, *args, **kwargs):
    start_value = time.perf_counter()
    return_value = input_function(*args, **kwargs)
    end_value = time.perf_counter()
    runtime_value = end_value - start_value
    # print(f"Finished executing {input_function.__name__} in {runtime_value} seconds")
    return return_value, runtime_value


def entropy(x):
    exp_x = np.exp(x)
    A = np.sum(exp_x, axis=-1)  # sum of exp(x_i)
    B = np.sum(x * exp_x, axis=-1)  # sum of x_i * exp(x_i)
    return np.log(A) - B / A


def generate_random_data():
    num_samples = 10
    p_test = np.random.rand(num_samples, 3)
    y_test = np.random.choice(3, num_samples)
    return p_test, y_test


##### metrics #####


def geifman_AURC(f_X, g, Y, plot=False):
    residuals = f_X.argmax(-1) != Y
    confidence = g(f_X)
    curve = []
    m = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    cov = len(temp1)
    acc = sum(temp1)
    curve.append((cov / m, acc / len(temp1)))
    for i in range(0, len(idx_sorted) - 1):
        cov = cov - 1
        acc = acc - residuals[idx_sorted[i]]
        curve.append((cov / m, acc / (m - i)))
    AUC = sum([a[1] for a in curve]) / len(curve)
    return AUC


def official_AURC(f_X, g, Y):
    "The RC curve is obtained by computing the risk of the coverage from the beginning of g(x) (most confident) to the end (least confident)."
    incorrect = f_X.argmax(-1) != Y  # instance-level mask
    g_X = g(f_X)
    idx_sorted = np.argsort(g_X)  # in ascending format; construct curve from right to left

    coverages, risks = [], []
    weights = (
        []
    )  # just forms some mask of points that were different/used for integration (x/N) --> with percentage of data captured

    # well-defined starting point: risk = 1-accuracy (loss), coverage=100%; threshold=0
    coverages.append(1)
    risks.append(incorrect.mean())

    # will keep these as intermediate absolute values to facilitate calculation
    N = len(idx_sorted)
    coverage = len(idx_sorted)
    error_sum = sum(incorrect[idx_sorted])

    tmp_weight = 0
    for tau in range(0, len(idx_sorted) - 1):  # each
        coverage = coverage - 1
        error_sum = error_sum - incorrect[idx_sorted[tau]]
        selective_risk = error_sum / (N - 1 - tau)
        tmp_weight += 1
        if tau == 0 or g_X[idx_sorted[tau]] != g_X[idx_sorted[tau - 1]]:  # unique or starting threshold
            coverages.append(coverage / N)
            risks.append(selective_risk)
            weights.append(tmp_weight / N)
            tmp_weight = 0

    # well-defined ending (if not already done): last known risk for 0 coverage (threshold=100%)
    if tmp_weight > 0:
        coverages.append(0)
        risks.append(risks[-1])
        weights.append(tmp_weight / N)  # rest of the data

    aurc = sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))])
    return aurc


def AURC_naive(f_X, g, Y):
    N = len(Y)
    outer_sum = 0
    for h in range(N):
        num = 0  # risk
        den = 0  # coverage
        for i in range(N):  # inner sums
            if i == h:
                continue
            loss = int(f_X[i].argmax(-1) != Y[i])
            indicator = g(f_X[i]) > g(f_X[h])
            num += loss * indicator  # empirical risk for h vs. i
            den += indicator  # empirical coverage for h vs. i

        if den == 0:  # no coverage, no risk?
            continue
        outer_sum += num / den  # empirical risk for h vs. all i
    return outer_sum / N  # , risks, coverages


def AURC_naive_alphas_ON2(f_X, g, Y):
    N = len(Y)
    indices_sorted = np.argsort(g(f_X))

    # assume all are sorted
    sorted_f_X = f_X[indices_sorted]
    sorted_y = Y[indices_sorted]
    losses = sorted_f_X.argmax(-1) != sorted_y

    final_sum = 0
    for i in range(N - 1):
        loss_sum = np.sum(losses[i + 1 :])
        final_sum += loss_sum / (N - i - 1)
    return final_sum / N


# loss = 0/1
def AURC_naive_alphas_ON(f_X, g, Y):
    N = len(Y)
    indices_sorted = np.argsort(-g(f_X))  # descending sort
    sorted_f_X = f_X[indices_sorted]
    sorted_y = Y[indices_sorted]
    final_sum = 0
    partial_loss = int(sorted_f_X[0].argmax(-1) != sorted_y[0])  # init by largest
    for i in range(1, N):
        final_sum += partial_loss / i
        partial_loss += int(sorted_f_X[i].argmax(-1) != sorted_y[i])
    return final_sum / N


IMPLEMENTATIONS = [
    ("geifman", geifman_AURC),
    ("jaeger 2023", official_AURC),
    ("naive", AURC_naive),
    ("naive_alphas", AURC_naive_alphas_ON),
]


if __name__ == "__main__":
    g = lambda x: -entropy(x)
    p_simp, y_simp = generate_random_data()
    naive = AURC_naive(p_simp, g, y_simp)
    print(naive)
    naive_alphas = AURC_naive_alphas_ON2(p_simp, g, y_simp)
    print(naive_alphas)
    naive_alphas = AURC_naive_alphas_ON(p_simp, g, y_simp)
    print(naive_alphas)

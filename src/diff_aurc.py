import numpy as np


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
            # print('Comparison for h={} has 0 denominator'.format(h)) #which means it is max?
            continue
        outer_sum += num / den  # empirical risk for h vs. all i
    return outer_sum / N  # , risks, coverages


def AURC_naive_alphas(f_X, g, Y):
    N = len(Y)
    indices_sorted = np.argsort(g(f_X))
    sorted_f_X = f_X[indices_sorted]
    sorted_y = Y[indices_sorted]
    losses = sorted_f_X.argmax(-1) != sorted_y

    final_sum = 0
    for i in range(N - 1):
        loss_sum = np.sum(losses[i+1:])
        final_sum += loss_sum / (N - i - 1)

    return final_sum / N


if __name__ == '__main__':
    g = lambda x: -entropy(x)
    p_simp, y_simp = generate_random_data()
    naive = AURC_naive(p_simp, g, y_simp)
    print(naive)
    naive_alphas = AURC_naive_alphas(p_simp, g, y_simp)
    print(naive_alphas)

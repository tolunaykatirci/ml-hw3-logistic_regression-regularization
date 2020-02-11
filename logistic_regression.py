# CSE4088 Homework #3
# Tolunay Katirci - 150115014

import math
import numpy as np
import random


# computes a random line and returns a and b params: y = ax + b
def random_line():
    x1 = random.uniform(-1, 1)  # random value in range [-1,1]
    y1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    y2 = random.uniform(-1, 1)

    w1 = y2 - y1
    w2 = x2 - x1
    a = w1 / w2
    b = y1 - a * x1

    w = np.array([b * w2, -w1, w2])
    return w


# sign of a point
def sign(x, w):
    y = np.dot(x, w)
    return 1 if y >= 0 else -1


# create random data
def random_data(w, N=100):
    x = np.array([np.array([1] + [random.uniform(-1, 1) for i in range(2)]) for i in range(N)])
    y = np.apply_along_axis(sign, 1, x, w)
    return (x, y)


def log_e(x1, y1, w):
    e = np.log(1.0 + math.exp(-y1 * np.dot(w, x1)))
    return e


def log_delta_e(x1, y1, w):
    delta_e = -(y1 * x1) / (1.0 + math.exp(y1 * np.dot(w, x1)))
    return delta_e


def parse(x, y, w):
    es = np.zeros(y.shape)
    for i in range(y.size):
        es[i] = log_e(x[i], y[i], w)
    return np.mean(es, axis=0)


def SGD(x, y, learning_rate=0.01):
    # initial values
    w = np.array([0, 0, 0])
    d = 1.0
    count = 0
    n = y.size
    w_prev = w
    E_list = []

    random_p = random.sample(range(n), n)

    while 0.01 < d:
        if len(random_p) == 0:
            count += 1
            d = np.sqrt(np.sum((w_prev - w) ** 2))  # diff norm
            if d < 0.01:
                # reached to expected w
                print("Epoch: %d, E_out: %.4f" % (count, (sum(E_list)/len(E_list))))
                return w, E_list, count

            w_prev = w
            random_p = random.sample(range(n), n)

        i = random_p.pop()
        E = log_e(x[i], y[i], w)
        E_list.append(E)

        # gradient descent
        g = log_delta_e(x[i], y[i], w)
        w = w - learning_rate*g

    return w, E_list, count


# sample size
N = 100
E_sum = 0.0
C_sum = 0.0

for i in range(N):

    print("Sample %s, " % (i+1), end='')
    w = random_line()
    (x, y) = random_data(w, N)
    # algorithm and results
    w, E_list, count = SGD(x, y, learning_rate=0.01)

    (x_out, y_out) = random_data(w, 500)
    E_out = parse(x_out, y_out, w)
    E_sum += E_out
    C_sum += count

# output
print("Average E_out: ", E_sum/N)
print("Average Epochs: ", C_sum/N)

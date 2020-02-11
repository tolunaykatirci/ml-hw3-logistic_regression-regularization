# CSE4088 Homework #3
# Tolunay Katirci - 150115014

import numpy as np
from scipy.optimize import fmin
from math import e


def ff(xx):
    return pow((xx[0] * pow(e, xx[1]) - 2 * xx[1] * pow(e, -xx[0])), 2)


def gradient_descent_algorithm(x, y, learning_rate=0.1, error_rate=0.000001):
    # f function
    f = lambda x, y: pow((x * pow(e, y) - 2 * y * pow(e, -x)), 2)

    # partial derivatives of f
    df_x = lambda x, y: 2 * (pow(e, y) + 2 * y * pow(e, -x)) * (x * pow(e, y) - 2 * y * pow(e, -x))
    df_y = lambda x, y: 2 * (x * pow(e, y) - 2 * y * pow(e, -x)) * (x * pow(e, y) - 2 * pow(e, -x))

    max_iterations = 10000  # maximum number of iterations
    current_x, current_y = x, y

    result = fmin(ff, np.array([1, 1]))
    min_f = f(result[0], result[1])

    print("The minimum value of E(u,v): ", min_f)
    print("u: %s, v: %s" % (result[0], result[1]))
    print()

    i = 0
    for i in range(max_iterations):
        i += 1
        prev_x, prev_y = current_x, current_y

        # gradient descent
        current_x = prev_x - learning_rate * df_x(prev_x, prev_y)
        current_y = prev_y - learning_rate * df_y(prev_x, prev_y)

        # check if achieved to expected error rate
        precision = f(current_x, current_y)
        if (precision - min_f) < error_rate:
            print("Achieved to expected error rate... ")
            break

    precision = f(current_x, current_y)

    # results
    print("The achieved minimum value of E(u,v): ", precision)
    print("u: %s, v: %s" % (current_x, current_y))
    print("Iteration: %s" % i)
    print("Current error rate: %s" % (precision - min_f))


# run algorithm
gradient_descent_algorithm(1, 1, error_rate=pow(10, -14))

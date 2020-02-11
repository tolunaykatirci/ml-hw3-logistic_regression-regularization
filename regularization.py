# CSE4088 Homework #3
# Tolunay Katirci - 150115014

import numpy as np


# linear regression
def linear_regression(training_x, training_y, test_x, test_y, lmbd=0, transformation_x=None, transformation_y=None):

    # transforms
    x = transform_x(transformation_x, training_x)
    y = transform_y(transformation_y, training_y)
    assert y.shape[0] == x.shape[0]

    d = x.shape[1]
    w = np.linalg.pinv(x.T.dot(x) + lmbd * np.eye(d)).dot(x.T).dot(y)

    # in-sample and out-of-sample errors
    training_error = error(training_x, training_y, w, transformation_x, transformation_y)
    test_error = error(test_x, test_y, w, transformation_x, transformation_y)

    # result
    print('training error: %f, test error: %f' % (training_error, test_error))


# transformation of x
def transform_x(transformation_x, x):
    if transformation_x is not None:
        for t in transformation_x:
            x = t(x)
    return x


# transformation  of y
def transform_y(transformation_y, y):
    if transformation_y is not None:
        for t in transformation_y:
            y = t(y)
    return y


# prediction
def predict(x, w, transformation_x):
    x = transform_x(transformation_x, x)
    assert w is not None
    assert w.shape[0] == x.shape[1]

    predicted = x.dot(w)
    predicted[predicted >= 0] = 1
    predicted[predicted < 0] = -1
    return predicted


# classification error
def error(x, y, w, transformation_x, transformation_y):
    predicted = predict(x, w, transformation_x)
    y = transform_y(transformation_y, y)
    return 1 - (y == predicted).sum() / predicted.size


def phi_transform(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    N = x1.size
    return np.vstack([np.ones(N),
                      x1,
                      x2,
                      x1 ** 2,
                      x2 ** 2,
                      x1 * x2,
                      np.abs(x1 - x2),
                      np.abs(x1 + x2)]).T


def add_bias(X):
    return np.vstack([np.ones(X.shape[0]), X.T]).T


# read data
training = np.loadtxt('data/in.dta')
test = np.loadtxt('data/out.dta')

training_set_X = training[:, 0:2]
training_set_Y = training[:, 2]

test_X = test[:, 0:2]
test_Y = test[:, 2]

# question 2
print("Linear Regression => ", end='')
linear_regression(training_set_X, training_set_Y, test_X, test_Y, transformation_x=(phi_transform, add_bias))

# question 3
k = -3
print("\nLinear Regression, k=-3 =>  ", end='')
linear_regression(training_set_X, training_set_Y, test_X, test_Y, lmbd=10 ** k, transformation_x=(phi_transform, add_bias))

# question 4
k = 3
print("\nLinear Regression, k=3 =>  ", end='')
linear_regression(training_set_X, training_set_Y, test_X, test_Y, lmbd=10 ** k, transformation_x=(phi_transform, add_bias))

# question 5-6
print()
for k in range(-2, 3):
    print("Linear Regression, k=%d =>  " % k, end='')
    linear_regression(training_set_X, training_set_Y, test_X, test_Y, lmbd=10 ** k, transformation_x=(phi_transform, add_bias))

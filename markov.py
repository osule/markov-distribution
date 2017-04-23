import numpy as np


def to_nparray(X):
    return np.array(X)

def meanNd(X):
    return (X[:, n].mean() for n in range(0, X.ndim))

def totalNd(X):
    return (X[:, n].sum() for n in range(0, X.ndim))

def shift_array(Xn):
    return np.delete(Xn, -1)

def shift_arrayNd(X):
    Xn = X[:, X.ndim - 1].copy()
    return shift_array(Xn)

def sum_and_mean(X):
    return X.sum(), X.mean()

def X1Xn(X):
    product = X[:, 0].copy()
    for n in range(1, X.ndim):
        product *= X[:, n]
    return product

def XX(Xn):
    return np.square(Xn)

def XXNd(X):
    return (XX(X[:, n]) for n in range(0, X.ndim))

def X0X1Nd(X):
    X0 = shift_arrayNd(X)
    X1 = X[1:, 0]

    return X0 * X1

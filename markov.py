import numpy as np
import math
import scipy.stats


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

def Round(X):
    return round(X, 2)

def B1(X):
    X0 = Round(X0X1Nd(X).mean())
    X1 = shift_arrayNd(X)
    X2 = (X[:, 0])
    X3 = Round(X1.mean()) * Round(X2.mean())
    X4 = X1 * X1
    X42 = Round(X4.mean())
    X5 = Round(X1.mean()) 
    X6 = X5 * X5
    numerator = X0 - X3
    denominator = X42 - X6
    return numerator / denominator

def B2(X):
    X0 = Round(X1Xn(X).mean())
    X1 = (X[:, 0])
    X2 = (X[:, 1])
    X3 = Round((X1 * X2).mean())
    X4 = Round(X1.mean() * X2.mean())
    X5 = Round((X1 * X1).mean())
    X7 = Round(X1.mean())
    X8 = Round(X7 * X7)
    numerator = X0 - X4
    denominator = X5 - X8
    return numerator / denominator

def SD(Xn):
    X1 = (Xn * Xn).mean()
    X2 = Xn.mean()
    X3 = X2 * X2
    X4 = X1 - X3
    return math.sqrt(X4)


def R1(X):
    X0 = shift_arrayNd(X)
    X1 = (X[:, 0])
    SD1 = SD(X0)
    SD2 = SD(X1)
    numerator = B1(X) * SD2
    denominator = SD1
    return numerator / denominator

def R2(X):
    X0 = shift_arrayNd(X)
    X1 = (X[:, 0])
    SD1 = SD(X0)
    SD2 = SD(X1)
    numerator = B2(X) * SD1
    denominator = SD2
    return numerator / denominator

def RandomGenerator(n):
    return np.random.uniform(00, 99, n)

def ZScore():
    X0 = RandomGenerator(2)
    Str = str(int(X0[0])) + str(int(X0[1]))
    X1 = int(Str, 10)
    X2 =  X1 / 10000.0
    X3 = scipy.stats.norm.ppf(X2)
    X4 = round(X3, 3)
    T = X4 if(X2 > 0.5) else (X4 * (-1))
    print(round(T, 3), X3)
    return T

def Sum(a, b):
    return a + b

def MarkovModel(X):
    Q1 = X[:, 0]
    Q2 = X[:, 1]
    b1 = B1(X)
    b2 = B2(X)
    X2 = X[:, 1]
    Q_i_1_J_1 = X2[len(X2) - 1]
    QJ_1 = X2.mean()
    T1 = ZScore()
    T2 = ZScore()
    SD1 = SD(shift_arrayNd(X))
    SD2 = SD(Q1)
    RC1 = math.sqrt(1 - (R1(X) * R1(X)))
    RC2 = math.sqrt(1 - (R2(X) * R2(X)))
    quarterlyFlow1 = Q1.mean() + (b1 *(Q_i_1_J_1 - QJ_1)) + (T1 * SD1 * RC1)
    quarterlyFlow2 = Q2.mean() + (b1 *(quarterlyFlow1 - QJ_1)) + (T2 * SD2 * RC2)
    #'{} {}'.format("Quarterly Streamline Flows for the first period", quarterlyFlow1)
    print('Quarterly Streamline Flows for the first period :' , quarterlyFlow1)
    print('Quarterly Streamline Flows for the Second period :' , quarterlyFlow2)
    

    
    


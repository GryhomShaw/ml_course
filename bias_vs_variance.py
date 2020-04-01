import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt


def load_data(path):
    path = './ex5/ex5data1.mat'
    data = sio.loadmat(path)
    return data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']


def show_data(X, y):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X, y, s=50, c='b')
    plt.show()


def show_res(X, y, theta):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X[:, 1:], y, s=50, c='b', label='Training data')
    pred = np.dot(X, theta)
    ax.plot(X[:, 1:], pred, label='Pred')
    plt.show()

def cost(theta, X, y):
    m = X.shape[0]
    pred = np.dot(X, theta)
    err = pred - np.squeeze(y)
    return (1/(2*m)) * np.sum(err * err)


def regularized_cost(theta, X, y, lamba = 1):
    m = X.shape[0]
    regularized_term = (lamba / (2*m)) * np.power(theta[1:], 2).sum()
    return cost(theta, X, y) + regularized_term


def gradient (theta, X, y):
    m = X.shape[0]
    pred = np.dot(X, theta)
    err = pred - np.squeeze(y)
    return (1 / m) * np.dot(X.T, err)


def regularized_gradient(theta, X, y, lamba=1):
    m = X.shape[0]
    regularized_term = theta.copy()
    regularized_term[0] = 0
    return gradient(theta, X, y) + (lamba / m) * regularized_term


def linear_regression(X, y, lamba=1):
    theta = np.ones(X.shape[1])
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y, lamba), method='TNC', jac=regularized_gradient)
    return res.get('x')


def prepare_ploy_data(*args, power=3):
    res = []
    for each in args:
        each = np.squeeze(each)
        data = np.array([np.power(each, p) for p in range(1, power+1)]).T
        data_norm = normalize_feature(data)
        res.append(np.insert(data_norm, 0, np.ones(data_norm.shape[0]), axis=1))
    return res


def normalize_feature(data):
    for col in range(data.shape[1]):
        mean = np.mean(data[:, col])
        std = np.std(data[:, col])
        data[:, col] = (data[:, col] - mean) / std
    return data


def plot_learning_curve(X, y, Xval, yval, lamba =0):
    train_cost, cv_cost = [], []
    m = X.shape[0]
    for i in range(1, m+1):
        res = linear_regression(X[:i, :], y[:i, :], lamba=lamba)
        train_cost.append(cost(res, X[:i, :], y[:i, :]))
        cv_cost.append((cost(res, Xval, yval)))
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(np.arange(1, X.shape[0] + 1), train_cost, label='train cost')
    ax.plot(np.arange(1, X.shape[0] + 1), cv_cost, label='val cost')
    plt.show()


if __name__ == '__main__':
    path = './ex5/ex5data1.mat'
    X, y, Xval, yval, Xtest, ytest = load_data(path)
    #print(X.shape, y.shape, Xval.shape, yval.shape, Xtest.shape, ytest.shape)
    X, Xval, Xtest = [np.insert(each, 0, np.ones(each.shape[0]), axis=1) for each in [X, Xval, Xtest]]
    #print(X.shape, y.shape, Xval.shape, yval.shape, Xtest.shape, ytest.shape)
    theta = np.ones(X.shape[1])
    print(cost(theta, X, y))
    print(gradient(theta, X, y))
    print(regularized_gradient(theta, X, y))
    final_theta = linear_regression(X, y)
    print(final_theta)
    #show_res(X, y, final_theta)
    '''
    使用不同规模的训练集进行训练
    '''
    plot_learning_curve(X, y, Xval, yval, 0)


    '''
    通过学习曲线可i元看出上述模型存在过拟合现象, 所以想通过创建多项式特征解决
    '''
    X, y, Xval, yval, Xtest, ytest = load_data(path)
    X_poly, Xval_poly, Xtest_poly =  prepare_ploy_data(X, Xval, Xtest, power=8)
    plot_learning_curve(X_poly, y, Xval_poly, yval, 0)
    '''
    此时会发现随着数据规模的逐渐增大, 偏差已经足够小， 而方差在降低到一定值之后开始变大，并且不断波动，说明存在过拟合现象
    可以进行L2正则化，增大lamda
    '''
    plot_learning_curve(X_poly, y, Xval_poly, yval, 1)
    plot_learning_curve(X_poly, y, Xval_poly, yval, 100)
    '''
    通过搜索得到最佳的正则化系数,通过图像可以看出， 在0.3左右是更佳的
    '''
    l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    train_cost = []
    cv_cost = []
    for lamba in l_candidate:
        res = linear_regression(X_poly, y, lamba)
        train_cost.append(cost(res, X_poly, y))
        cv_cost.append(cost(res, Xval_poly, yval))
    plt.plot(l_candidate, train_cost, label = "Train_cost")
    plt.plot(l_candidate, cv_cost, label="cv_cost")
    plt.show()



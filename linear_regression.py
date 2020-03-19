import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data(data):
    data.insert(0, 'Ones', 1)
    cols = data.shape[1]
    X = data.iloc[:, :cols - 1]
    y = data.iloc[:, cols - 1:]
    X = X.values
    y = y.values
    return X, y


def linear_function(data,theta):
    #data m*n
    return np.dot(data,theta)


def cost_function(predict, y):
    m = predict.shape[0]
    error = np.sum(np.power(predict - y, 2))
    return error / (2*m)


def gradientDescent(X, y, theta, alpha, iters):
    # X m*n
    # y = n*1
    # theta = N * 1
    cost = np.zeros(iters)
    m = X.shape[0]
    print(theta.shape)
    for i in range(iters):
        #print(theta)
        error = linear_function(X,theta) - y
        for each in range(theta.shape[0]):
            temp = error *  X[:, each : each+1]
            theta[each][0] = theta[each][0] - (alpha / m) * np.sum(temp)
        cost[i] = cost_function(linear_function(X, theta), y)
    return theta, cost


def plot_lr(data,theta):
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = theta[0][0] + x * theta[1][0]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()


def plot_loss(iters, cost):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel("Iterations")
    ax.set_ylabel('loss')
    ax.set_title('Error VS. Training Epoch')
    plt.show()


def signle_linear_regression():
    path = './ex1/ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    # print(data.head())
    data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
    plt.show()
    #print(data.describe())
    X, y = get_data(data)
    theta = np.array([0.0, 0.0]).reshape(2, 1)
    pred = linear_function(X, theta)
    loss =  cost_function(pred, y)
    print("loss: {}".format(loss))
    theta, cost = gradientDescent(X, y, theta, 0.01, 1000)
    print(theta)
    plot_lr(data, theta)
    plot_loss(1000, cost)


def multi_linear_regression():
    path = './ex1/ex1data2.txt'
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    ##特征归一化
    data = (data - data.mean()) / data.std()
    X, y = get_data(data)
    theta = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
    theta, cost = gradientDescent(X, y, theta, 0.01, 1000)
    plot_loss(1000, cost)
    res = normalEqn(X, y)
    print(res, theta)


def normalEqn(X, y):

    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    return theta


if __name__ == '__main__':
    #signle_linear_regression()
    multi_linear_regression()

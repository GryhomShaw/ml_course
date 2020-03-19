import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmod(z):
    """
    :param z: m*1
    :return: m*1
    """
    return 1.0 / (1.0 + np.exp(-z))


def lr_func(X, theta):
    """
    :param X: m*(n+1)
    :param theta: (n+1) * 1
    :return: m *1
    """
    return sigmod(np.dot(X, theta))


def loss(X, theta, y):
    pred = lr_func(X, theta)
    m = pred.shape[0]
    #print(pred)
    '''
    for i in range(pred.shape[0]):
        res += np.log(pred[i][0]) if y[i][0] == 1 else np.log(1 - pred[i][0])
    '''
    first = np.sum(y * np.log(pred))
    second = np.sum((1-y) * np.log(1 - pred))
    return -1*(first + second) / m


def regularized_loss(X, theta, y, lamba):
    m = X.shape[0]
    reg = (lamba / 2*m )*np.sum(np.power(theta[1:],2))
    return loss(X, theta, y) + reg


def predict(theta, X, threshold):
    pred = lr_func(X, theta)
    return (pred >= threshold).astype(np.int)


def gradient(X, theta, y, iters, alpha, weight_decay = 0):
    """
    :param X: m * n+1
    :param theta: n+1 * 1
    :param y: m * 1
    :param iters:
    :return: n+1 * 1
    """
    m = X.shape[0]
    grad = np.zeros([iters, X.shape[1]])
    for i in range(iters):
        pred = lr_func(X, theta)
        error = pred - y
        '''
        trivial!!!
        for j in range(theta.shape[0]):
            grad[i][j] = np.sum(error * X[:, j:j+1]) / m
            theta[j][0] = theta[j][0] - (alpha / m) * np.sum(error * X[:, j:j+1])
        '''
        weight = np.concatenate([np.array([0]), np.array([weight_decay] * (X.shape[1] - 1))])  #bias不需要weight_decay
        grad[i] = (np.dot(X.T, error) / m).reshape(X.shape[1]) + np.squeeze(weight*theta)
        theta = (1 - alpha*weight)*theta - (alpha / m) * np.dot(X.T, error)
    return theta, grad


def get_data(data):
    data.insert(0, 'Ones', 1)
    col = data.shape[1]
    X = data.iloc[:, : col-1]
    y = data.iloc[:, col-1]
    X = X.values
    y = y.values
    return X, y


def plot_data1(data):
    pos = data[data['Admitted'].isin([1])]
    neg = data[data['Admitted'].isin([0])]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(pos['Exam1'], pos['Exam2'], s=50, c='b', marker='o', label='Admitted')
    ax.scatter(neg['Exam1'], neg['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
    ax.legend()
    plt.show()


def plot_data2(data):
    pos = data[data['Accepted'].isin([1])]
    neg = data[data['Accepted'].isin([0])]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(pos['Test_1'], pos['Test_2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(neg['Test_1'], neg['Test_2'], s=50, c='r', marker='x', label='Not Accepted')
    ax.legend()
    plt.show()


def featrue_mapping(x1, x2, degree):
    data = {}
    for i in range(0, degree+1):
        for j in range(0,i+1):
            key = "F{}{}".format(i-j, j)
            data[key] = np.power(x1, i-j) * np.power(x2, j)
    return pd.DataFrame(data)


if __name__ == '__main__':
    '''
    data1_path = './ex2/ex2data1.txt'
    data = pd.read_csv(data1_path, header=None, names=['Exam1', 'Exam2', 'Admitted'])
    X, y = get_data(data)
    theta = np.array([0.0, 0.0, 0.0])
    print(X.shape, y.shape, theta.shape)
    print(loss(X, theta, y))
    theta, _ = gradient(X, theta, y, 2000000, 0.003)
    #theta = np.array([-25.1613186 , 0.20623159, 0.20147149]).reshape(3,1)
    print(theta)
    print(loss(X, theta, y))
    res = predict(theta, X, 0.5)
    acc= np.sum(np.squeeze(res) == np.squeeze(y)) / res.shape[0]
    print(acc)
    '''
    #############正则化逻辑回归#########
    data2_path = './ex2/ex2data2.txt'
    data = pd.read_csv(data2_path, header=None, names=['Test_1', 'Test_2', 'Accepted'])
    #print(data.head())
    plot_data2(data)
    x1 = np.array(data['Test_1'])
    x2 = np.array(data['Test_2'])
    #print(x1.shape, x2.shape)
    new_data = featrue_mapping(x1, x2, 6)
   # print(new_data.head())
    X = new_data.values
    cols = data.shape[1]
    y = data.iloc[:, cols-1]
    y = y.values

    theta = np.zeros(28)
    print(X.shape, y.shape, theta.shape)
    loss = regularized_loss(X, theta, y, 1)
    print(loss)
    lr = 0.01
    weight_decay =1.0 / X.shape[0]
    theta, grad = gradient(X, theta, y, 400000, lr, weight_decay)
    print(theta)
    res = predict(theta, X, 0.5)
    acc = np.sum(np.squeeze(res) == np.squeeze(y)) / res.shape[0]
    print(acc)








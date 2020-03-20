import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as opt
import matplotlib
from sklearn.metrics import classification_report

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def lr_func(X, theta):
    #print(X.shape, theta.shape)
    return sigmoid(np.dot(X, theta))

def cost(theta, X, y):
    """
    :param X: [m,n]
    :param thrta: n*
    :param y: n
    :return: 1
    """
    pred = lr_func(X, theta)
    return -1 * np.mean(y* np.log(pred) + (1-y) * np.log(1-pred))


def regularized_cost(theta, X, y, lamba=1):
    m = X.shape[0]
    #print(X.shape, theta.shape, y.shape)
    regularized_term = (lamba / (2*m)) * np.power(theta[1:],2).sum()
    return cost(theta, X, y) + regularized_term


def gradient(theta, X,  y):
    return (1 / len(X)) * np.dot(X.T, (lr_func(X, theta) - y))


def regularized_gradient(theta, X,  y, lamba=1):
    regularized_iterm = (lamba / X.shape[0]) * theta[1:]
    regularized_iterm = np.concatenate([np.array([0]), regularized_iterm])
    return gradient(theta, X, y) + regularized_iterm




def load_data(path, transpose = True):
    data = sio.loadmat(path)
    y = np.squeeze(data.get('y'))
    X = data.get('X')
    if transpose:
        X = np.array([ im.reshape((20,20)).T for im in X])
        X = np.array([im.reshape(400) for im in X])
    return X, y


def plot_img(image):
    fig, ax = plt.subplots(figsize = (1,1))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))  # just get rid of ticks
    plt.yticks(np.array([]))
    plt.show()


def plot_100_img(image):
    size = image.shape[1]
    sample_idx = np.random.choice(np.arange(image.shape[0]),100) #随机采样100个
    sample_images = image[sample_idx,:]

    fig, ax_arrays = plt.subplots(nrows=10, ncols=10, sharey='all', sharex='all', figsize=(8, 8))
    for i in range(10):
        for j in range(10):
            ax_arrays[i][j].matshow(sample_images[i*10+j].reshape(size, size), cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()


def data_visuailzation(path):
    X, y = load_data(path)
    print(X.shape, y.shape)
    plot_img(X[823, :, :])
    plot_100_img(X)



def labels_one_hot(labels,k):
    """
    :param labels: 5000
    :param k: 10
    :return:
    """
    res = []
    for each_label in range(1, k+1):
        res.append((labels == each_label).astype(int))
    res = [res[-1]] + res[:-1]
    return np.array(res)


def logistic_regression(X, y, l=1):
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y, l), method='TNC',
                       jac=regularized_gradient)
    #print(res)
    final_theta = res.x
    return final_theta


def predict(X, theta, threshaold=0.5):
    pred = lr_func(X, theta)
    return (pred >= threshaold).astype(np.int)


def nn_forward(X, theta1, theta2):
    """
    :param X: [5000,401]
    :param theta1: [25 * 401]
    :param theta2: [10,26]
    :return: [5000,10]
    """
    z2 = np.dot(X, theta1.T) # 5000 25 第二层的输入
    z2 = np.insert(z2, 0, values= np.ones(z2.shape[0]), axis=1)
    a2 = sigmoid(z2) #第二层的输出 5000 26
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    return a3


if __name__ == '__main__':
    path = './ex3/ex3data1.mat'
    '''
        data_prepare
    '''
    # X, y = load_data(path)
    # label = y
    # X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
    # y = labels_one_hot(y, 10)
    '''
        single classification
    '''
    # t0 =logistic_regression(X, y[0])
    # #print(t0)
    # print("ACC: {}".format(np.mean(predict(X,t0) == y[0])))
    '''
        multi classification
    '''
    # theta_array = []
    # for k in range(y.shape[0]):
    #     theta_array.append(logistic_regression(X, y[k]))
    # theta_array = np.array(theta_array)
    # print(theta_array.shape)
    # prob_matrix = sigmoid(np.dot(X, theta_array.T)) ## (5000, 401) (401, 10) -> (5000, 10)
    # np.set_printoptions(suppress=True)
    # y_pred = np.argmax(prob_matrix,axis = 1)
    # y_ans = label
    # y_ans[y_ans==10] = 0
    # print(classification_report(y_ans, y_pred))
    '''
        neturral network forward
    '''
    weight_path = './ex3/ex3weights.mat'
    weights = sio.loadmat(weight_path)
    theta1 = weights['Theta1']
    theta2 = weights['Theta2']
    print(theta1.shape, theta2.shape)
    X, y = load_data(path, transpose=False)
    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
    print(X.shape, y.shape)
    prob = nn_forward(X, theta1, theta2)
    y_pred = np.argmax(prob, axis=1) + 1
    print(classification_report(y_pred, y))



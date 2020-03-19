import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib
#from sklearn.metrics import calssification_report


def load_data(path, transpose = True):
    data = sio.loadmat(path)
    y = np.squeeze(data.get('y'))
    X = data.get('X')
    if transpose:
        X = np.array([ im.reshape(20,20).T for im in X])
    else:
        X = np.array([im.reshape(400) for im in X])

    return X, y


def plot_img(image):
    fig, ax = plt.subplots(figsize = (1,1))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))  # just get rid of ticks
    plt.yticks(np.array([]))
    plt.show()

if __name__ == '__main__':
    path = './ex3/ex3data1.mat'
    X, y = load_data(path)
    print(X.shape, y.shape)
    plot_img(X[823, :, :])


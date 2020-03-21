import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as opt


class NN:
    def __init__(self, layers, labels):
        self.layers = layers
        self.l_nums = len(layers)
        self.weight_shape = [[self.layers[i], self.layers[i-1]+1] for i in range(1, self.l_nums)]
        self. weights = self.init_weights()
        self.z_array = []
        self.a_array = []
        self.labels = labels

    def init_weights(self):
        weights = []
        for each in self.weight_shape:
            weights.append(np.random.standard_normal(size=each))
        return np.array(weights)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self):
        if len(self.a_array) == 0:
            print('No output!')
            return
        out = self.a_array[-1]
        assert out.shape == self.labels.shape, print('Shape error')
        m = out.shape[0]
        log_func = -1 *(self.labels * np.log(out) + (1 - self.labels) * np.log(1 - out))
        return log_func.sum() / m

    def regularized_cost(self, lamba=1):
        regularized_term = 0.0
        m = self.a_array[-1].shape[0]
        for each_weight in self.weights:
            regularized_term += np.power(each_weight[:,1:],2).sum()
        return self.cost() + (lamba / (2*m)) * regularized_term


    def forward(self, x, weights=None):
        if weights is not None:
            self.weights = weights
        self.a_array = []
        self.z_array = []
        a1 = np.insert(x, 0, values=np.ones(x.shape[0]), axis=1)
        self.a_array.append(a1)
        for i in range(1, self.l_nums):
            z = np.dot(self.a_array[i-1], self.weights[i-1].T)
            self.z_array.append(z)
            a_out = self.sigmoid(z)
            if i is not self.l_nums - 1:
                a = np.insert(a_out, 0, values=np.ones(a_out.shape[0]), axis=1)
                self.a_array.append(a)
            else:
                self.a_array.append(a_out)
        return self.a_array[-1]


def load_data(path, transpose=True):
    data = sio.loadmat(path)
    X = np.squeeze(data.get('X'))
    y = data.get('y')
    if transpose:
        X = np.array([(im.reshape(20,20).T).reshape(400) for im in X])
    return X, y


def plot_100_image(images):
    m = images.shape[0]
    sample_idx = np.random.choice(np.arange(m), 100)
    sample_images = images[sample_idx]
    fig, axs = plt.subplots(ncols=10, nrows=10, sharey='all', sharex='all', figsize=(8,8))
    for i in range(10):
        for j in range(10):
            axs[i][j].matshow(sample_images[i*10 + j].reshape(20,20), cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()


def labels_to_onehot(labels,k):
    one_hot = []
    for each_label in labels:
        temp = np.zeros(k)
        temp[each_label-1] = 1
        one_hot.append(temp)
    return np.array(one_hot)


if __name__ == '__main__':
    path = './ex4/ex4data1.mat'
    weights_path = './ex4/ex4weights.mat'
    X, y = load_data(path)
    print(X.shape, y.shape)
    #plot_100_image(X)
    X_raw, y_raw = load_data(path, transpose=False)
   # print(y_raw)
    y = labels_to_onehot(y_raw, 10)
    data = sio.loadmat(weights_path)
    weights = [data['Theta1'], data['Theta2']]
    layers = [400, 25, 10]
    nn = NN(layers, y)
   # print(nn.weight_shape)
    pred_matrix = nn.forward(X_raw, weights)
    print(pred_matrix)
    loss = nn.regularized_cost()
    print(loss)






import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio


def data_visual(data_path):
    mat = sio.loadmat(data_path)
    data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    print(data.head())
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(data.X1, data.X2, c='b')
    ax.set_title(data_path)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    plt.show()


def random_init(data, k):
    index = np.random.choice(data.shape[0], k)
    return data[index]


def _find_cluster(x, centers):
    '''
    :param x: n 表示n维特征
    :param centers: (k,n) k个质心， n维特征
    :return: k 距离低k个最近
    '''

    distances = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=centers - x)

    return np.argmin(distances)


def assign_cluster(data, centers):
    '''
    return np.apply_along_axis(lambda x :_find_cluster(x, centers), axis=1, arr=data)
    '''
    m, n = data.shape
    idx = np.zeros(m)
    for i, each_data in enumerate(data):
        dis = np.linalg.norm(centers - each_data, axis=1)
        idx[i] = np.argmin(dis)
    return idx


def new_centers (data, idx, k):
    n = data.shape[1]
    centeroids = np.zeros([k, n])
    for each in range(k):
        index = np.where(idx == each)[0]
        #print(type(index), index)
        centeroids[each, :] = np.mean(data[index, :], axis=0)
    return centeroids


def kemeans(data, initial_centeroids, max_iters):
    centers = initial_centeroids
    k = initial_centeroids.shape[0]
    idx = np.zeros(data.shape[0])
    for each_eapoch in range(max_iters):
        idx = assign_cluster(data, centers)
        centers = new_centers(data, idx, k)
    return idx, centers


if __name__ == '__main__':
    data_path = './ex7/ex7data2.mat'
    data_visual(data_path)
    mat = sio.loadmat(data_path)
    data = mat.get('X')
    initial_centeroids = random_init(data, 3)
    idx, centers = kemeans(data, initial_centeroids, 10)
    cluster1 = data[np.where(idx == 0)[0], :]
    cluster2 = data[np.where(idx == 1)[0], :]
    cluster3 = data[np.where(idx == 2)[0], :]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(cluster1[:, 0], cluster1[:, 1], c='r', label='Cluster 1')
    ax.scatter(cluster2[:, 0], cluster2[:, 1], c='g', label='Cluster 2')
    ax.scatter(cluster3[:, 0], cluster3[:, 1], c='b', label='Cluster 3')
    plt.show()





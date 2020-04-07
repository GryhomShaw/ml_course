import numpy as np
import pandas as pd
import sklearn.svm
import scipy.io as sio
import matplotlib.pyplot as plt


def data_visualization(data):
    fix, ax = plt.subplots(figsize=(8,6))
    ax.scatter(data[data['y'] == 1]['X1'], data[data['y'] == 1]['X2'], s=50, c='r')
    ax.scatter(data[data['y'] == 0]['X1'], data[data['y'] == 0]['X2'], s=50, c='b')
    plt.show()


def pred_visualization(data, svm_calss):
    fix, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data['X1'], data['X2'], s=50, c=data[svm_calss], cmap='RdBu')
    plt.show()


if __name__ == '__main__':
    path = './ex6/ex6data1.mat'
    mat = sio.loadmat(path)
    print(mat.keys())
    data = pd.DataFrame(mat['X'], columns=['X1', 'X2'])
    data['y'] = mat['y']
    print(data.head())
    data_visualization(data)
    '''
        c=1时，使用正则化程度很大，对离群点数据不敏感
    '''
    svc1 = sklearn.svm.LinearSVC(C=1, loss='hinge')
    svc1.fit(data[['X1', 'X2']], data['y'])
    print(svc1.score(data[['X1', 'X2']], data['y']))
    data['SVM1'] = svc1.decision_function(data[['X1', 'X2']])
    pred_visualization(data, 'SVM1')
    '''
        c=100时，使用正则化程度很小，离群点数据敏感
    '''
    svc100 = sklearn.svm.LinearSVC(C=100, loss='hinge')
    svc100.fit(data[['X1', 'X2']], data['y'])
    print(svc100.score(data[['X1', 'X2']], data['y']))
    data['SVM100'] = svc100.decision_function(data[['X1', 'X2']])
    pred_visualization(data, 'SVM100')
    print(data.head())



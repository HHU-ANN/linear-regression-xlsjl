# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np
def ridge(data):
    x, y = read_data()
    alpha = 0.5
    I = np.identity(x.shape[1])
    coefficient_matrix = x.T.dot(x) + alpha * I
    w = np.linalg.solve(coefficient_matrix, x.T.dot(y))
    weight = w
    return data @ weight


def lasso(data):
    x, y = read_data()
    weights = np.random.randn(6)
    Lambda = 0.5
    learning_rate = 0.0007 #6

    for i in range(2000):
        y_pred = x @ weights
        error = y - y_pred

        for j in range(1, 6):
            d_regularization = Lambda * np.sign(weights[j])
            gradient = - (x[:, j] @ error) / 6 + d_regularization
            weights[j] -= learning_rate * gradient / np.max(np.abs(gradient))

        gradient = - error.mean()
        weights[0] -= learning_rate * gradient

    return data @ weights

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
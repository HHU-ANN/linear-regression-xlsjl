# 最终在main函数中传入一个维度为6的numpy数组，输出预测值
import numpy as np
import os


# try:
#   import numpy as np
# except ImportError as e:
#    os.system("sudo pip3 install numpy")
#    import numpy as np


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
    learning_rate = 0.0007

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


def read_data(path='C:/Users/10438/linear-regression-xlsjl/data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y


features = np.array([
    [2.0133330e+03, 1.6400000e+01, 2.8932480e+02, 5.0000000e+00, 2.4982030e+01, 1.2154348e+02],
    [2.0126670e+03, 2.3000000e+01, 1.3099450e+02, 6.0000000e+00, 2.4956630e+01, 1.2153765e+02],
    [2.0131670e+03, 1.9000000e+00, 3.7213860e+02, 7.0000000e+00, 2.4972930e+01, 1.2154026e+02],
    [2.0130000e+03, 5.2000000e+00, 2.4089930e+03, 0.0000000e+00, 2.4955050e+01, 1.2155964e+02],
    [2.0134170e+03, 1.8500000e+01, 2.1757440e+03, 3.0000000e+00, 2.4963300e+01, 1.2151243e+02],
    [2.0130000e+03, 1.3700000e+01, 4.0820150e+03, 0.0000000e+00, 2.4941550e+01, 1.2150381e+02],
    [2.0126670e+03, 5.6000000e+00, 9.0456060e+01, 9.0000000e+00, 2.4974330e+01, 1.2154310e+02],
    [2.0132500e+03, 1.8800000e+01, 3.9096960e+02, 7.0000000e+00, 2.4979230e+01, 1.2153986e+02],
    [2.0130000e+03, 8.1000000e+00, 1.0481010e+02, 5.0000000e+00, 2.4966740e+01, 1.2154067e+02],
    [2.0135000e+03, 6.5000000e+00, 9.0456060e+01, 9.0000000e+00, 2.4974330e+01, 1.2154310e+02]
])
print(ridge(features))
print(lasso(features))

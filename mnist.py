import numpy as np
import json

# 全局参数
# 训练集比例
TRAIN_RATIO = 0.8

# 迭代次数
ITER_NUM = 2000

# 学习率
LR = 0.01

# 数据集路径
DATA_FILE_PATH='../data/housing/housing.data'

def load_data(file_name=DATA_FILE_PATH):
    data_file = file_name
    data = np.fromfile(data_file, sep=' ')

    # data包含了所有的数据，每一条数据包括13个特征值及一个真值
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]

    feature_num = len(feature_names)
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 切分数据集,数据集偏移量
    offset = int(data.shape[0] * TRAIN_RATIO)
    training_data = data[:offset]

    # 数据归一化，都归一化到0-1
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), (training_data.sum(axis=0) / training_data.shape[0])
    for i in range(feature_num):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

class Network(object):
    def __init__(self, num_of_weights):
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0

    def forward(self, x):
        return np.dot(x, self.w) + self.b

    def loss(self, z, y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost

    """
    gradient = xij(zi - yi)/N

    """
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = np.mean(z - y)

        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, lr=0.01):
        self.w = self.w - lr * gradient_w
        self.b = self.b - lr * gradient_b

    def train(self, x, y, iter=100, lr=0.01):
        losses = []
        for i in range(iter):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, lr)
            losses.append(L)

            # print every 10 times
            if i % 10 == 0:
                print("iter {0}, loss:{1}".format(i, L))
        
        return losses

if __name__ == "__main__":
    # 获取数据
    training_data, test_data = load_data()
    x = training_data[:, :-1]
    y = training_data[:, -1:]

    net = Network(13)
    losses = net.train(x, y, iter=ITER_NUM, lr=LR)
    
    import matplotlib.pyplot as plt

    plot_x = np.arange(ITER_NUM)
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()

# %matplotlib inline
import sys
from IPython import display
from mxnet import autograd, gluon, np, npx
from d2l import mxnet as d2l

npx.set_np()
d2l.use_svg_display()

# 批量大小
batch_size = 256
# 加载训练数据集和测试数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# for X, y in test_iter:
#     print(X)
#     print(y)
#     break

# 初始化模型参数
# 这将构成一个784x10的权重矩阵
num_inputs = 784 # 图像大小为28x28, 每个像素一个特征
num_outputs = 10 # 10个类别
W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
W.attach_grad()
b.attach_grad()

# 定义softmax操作
# softmax的作用是将一组输出转换为概率，和为1, 通常用在多分类问题
# 这个函数可以一次处理一批样本，假设一个样本得到的输出为[o1 o2 o3]：
# [o1 o2 o3]       [exp(o1) exp(o2) exp(o3)]       [exp(o1) + exp(o2) + exp(o3)] [sum]     [exp(o1)/sum exp(o2)/sum exp(o3)/sum]
# [o1 o2 o3] exp-> [exp(o1) exp(o2) exp(o3)] sum-> [exp(o1) + exp(o2) + exp(o3)] [sum] /-> [exp(o1)/sum exp(o2)/sum exp(o3)/sum]
# [o1 o2 o3]       [exp(o1) exp(o2) exp(o3)]       [exp(o1) + exp(o2) + exp(o3)] [sum]     [exp(o1)/sum exp(o2)/sum exp(o3)/sum]
def softmax(O):
    O_exp = np.exp(O)
    partition = O_exp.sum(1, keepdims=True)
    return O_exp / partition

# 定义模型 softmax(XW + b)
# 这里的X是一批具有N个样本的数据, 每个样本具有784个特征
# reshape后得到一个N x 784的矩阵，再乘以权重矩阵784 x 10
# 得到一个N x 10的结果矩阵, 然后求softmax
def net(X):
    # print(f'X.shape: {X.reshape((-1, W.shape[0])).shape} - W.shape: {W.shape}')
    return softmax(np.dot(X.reshape((-1, W.shape[0])), W) + b)

# 定义损失函数
# 这里用的是交叉熵损失函数
# y是y_hat正确预测项的索引, 即y_hat是模型输出，y是标签
def cross_entropy(y_hat, y):
    return -np.log(y_hat[range(len(y_hat)), y])

# 分类精度, 即正确预测数量与总预测数量之比
# accuracy计算预测正确的数量
def accuracy(y_hat, y):  #@save
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 这里获取的是最大值的索引
        # 最大值的索引与y的索引相同说明预测是正确的
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.astype(y.dtype) == y
    return float(cmp.astype(y.dtype).sum())

# 计算计算数据集在模型net上的精度
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# test
print(evaluate_accuracy(net, test_iter))

# 训练一个迭代周期
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    return metric[0] / metric[2], metric[1] / metric[2]

# 用于绘制
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

# 训练，迭代num_epochs个周期
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics

        # print(W)
        # print(f'train_loss: {train_loss} train_acc: {train_acc} test_acc: {test_acc}')
        assert train_loss < 0.8, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

# 执行训练
num_epochs = 10
print(f'training 迭代周期: {num_epochs} 学习率: {lr}...')
# print(W)
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

# 预测
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

# print("predict...")
predict_ch3(net, test_iter)

from mxnet import autograd, gluon, init, npx, np
from d2l import mxnet as d2l

npx.set_np()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数, num_hiddens为隐藏层输出，它将作为输出层的输入
# 784 -> 256 -> 10
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()

# 激活函数
def relu(X):
    return np.maximum(X, 0)

# 模型
def net(X):
    X = X.reshape((-1, num_inputs))
    hide = np.dot(X, W1) + b1
    H = relu(hide)
    O = np.dot(H, W2) + b2
    # print(hide)
    # print(H)
    # print(O)
    return O

# 损失函数
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# 训练一个迭代周期
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 训练损失总和、训练准确度总和、样本数
    metric = d2l.Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.size)
    return metric[0] / metric[2], metric[1] / metric[2]

# 训练，迭代num_epochs个周期
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = d2l.evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics

        # print(f'train_loss: {train_loss} train_acc: {train_acc} test_acc: {test_acc}')
        # assert train_loss < 0.8, train_loss
        # assert train_acc <= 1 and train_acc > 0.7, train_acc
        # assert test_acc <= 1 and test_acc > 0.7, test_acc

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

# 训练
num_epochs, lr = 10, 0.1
train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))

# 预测
predict_ch3(net, test_iter)

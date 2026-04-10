# %matplotlib inline
import sys
from IPython import display
from mxnet import autograd, gluon, np, npx
from d2l import mxnet as d2l

npx.set_np()
d2l.use_svg_display()

mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)

print(len(mnist_train), len(mnist_test))
print(mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.asnumpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X, y = mnist_train[:18]

print(X.shape)
print(y)
show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y));

batch_size = 256

def get_dataloader_workers():  #@save
    """在非Windows的平台上，使用4个进程来读取数据"""
    # return 0 if sys.platform.startswith('win') else 4
    return 0

# 通过ToTensor实例将图像数据从uint8格式变换成32位浮点数格式，并除以255使得所有像素的数值
# 均在0～1之间
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size, shuffle=True,
                                   num_workers=get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
# train_iter, test_iter = load_data_fashion_mnist(32)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    print(X[0].shape, X[0].dtype)
    break

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
    print(f'X.shape: {X.reshape((-1, W.shape[0])).shape} - W.shape: {W.shape}')
    return softmax(np.dot(X.reshape((-1, W.shape[0])), W) + b)

# 定义损失函数
# 这里用的是交叉熵损失函数
# y是y_hat正确预测项的索引, 即y_hat是模型输出，y是标签
def cross_entropy(y_hat, y):
    return -np.log(y_hat[range(len(y_hat)), y])

# 分类精度, 即正确预测数量与总预测数量之比
# accuracy计算预测正确的数量
def accuracy(y_hat, y):
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

# %matplotlib inline
from mxnet import autograd, gluon, np, npx
from d2l import mxnet as d2l

npx.set_np()

# 一般来说，k个GPU并行训练过程如下：
# 在任何一次训练迭代中，给定的随机的小批量样本都将被分成k个部分，并均匀地分配到GPU上；
# 每个GPU根据分配给它的小批量子集，计算模型参数的损失和梯度；
# 将k个GPU中的局部梯度聚合，以获得当前小批量的随机梯度；
# 聚合梯度被重新分发到每个GPU中；
# 每个GPU使用这个小批量随机梯度，来更新它所维护的完整的模型参数集。

# LeNet(稍作修改)多gpu实现
# 初始化模型参数
scale = 0.01
# 3x3卷积层权重, 卷积核张量shape:
# co x ci x kh x kw (输出通道数 x 输入通道数 x 卷积核窗口高 x 卷积核窗口宽)
W1 = np.random.normal(scale=scale, size=(20, 1, 3, 3))
b1 = np.zeros(20)
# 5x5卷积层权重, 输入通道数等于上一层的输出通道数20，当前输出通道数为50
W2 = np.random.normal(scale=scale, size=(50, 20, 5, 5))
b2 = np.zeros(50)
# 全链接层权重，输入维度800，输出维度128
W3 = np.random.normal(scale=scale, size=(800, 128))
b3 = np.zeros(128)
# 全链接层权重，输入维度128，输出维度10
W4 = np.random.normal(scale=scale, size=(128, 10))
b4 = np.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 定义模型
def lenet(X, params):
    # 卷积层1
    h1_conv = npx.convolution(data=X, weight=params[0], bias=params[1],
                              kernel=(3, 3), num_filter=20)
    # 激活函数
    h1_activation = npx.relu(h1_conv)
    # 平均汇聚层
    h1 = npx.pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    # 卷积层2
    h2_conv = npx.convolution(data=h1, weight=params[2], bias=params[3],
                              kernel=(5, 5), num_filter=50)
    # 激活函数
    h2_activation = npx.relu(h2_conv)
    # 平均汇聚层
    h2 = npx.pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    # 展开成(batch_size, 通道数xhxw), 因为后面要送入全连接层
    h2 = h2.reshape(h2.shape[0], -1)
    # 全连接层
    h3_linear = np.dot(h2, params[4]) + params[5]
    h3 = npx.relu(h3_linear)
    # 最后一个全连接层，也就是输出
    y_hat = np.dot(h3, params[6]) + params[7]
    return y_hat

# test
X = np.zeros((2, 1, 28, 28), dtype=np.float32)  # (N, C, W, H)
Y = lenet(X, params)
print(Y)


# 交叉熵损失函数
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# 12.5.4. 数据同步
# 对于高效的多GPU训练，我们需要两个基本操作。 首先，我们需要向多个设备分发参数并附加梯度（get_params）。 如果没有参数，就不可能在GPU上评估网络。 第二，需要跨多个设备对参数求和，也就是说，需要一个allreduce函数。
# 将所有参数考到同一个设备
def get_params(params, device):
    new_params = [p.copyto(device) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params

# test
new_params = get_params(params, d2l.try_gpu(0))
print('b1权重: ', new_params[1])
print('b1梯度: ', new_params[1].grad)

# 假设现在有一个向量分布在多个GPU上，下面的allreduce函数将所有向量相加，并将结果广播给所有GPU。 请注意，我们需要将数据复制到累积结果的设备，才能使函数正常工作。
def allreduce(data):
    # 所有数据发到gpu0上累计
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].ctx)
    # 累加结果广播到所有gpu
    for i in range(1, len(data)):
        data[0].copyto(data[i])

# test
data = [np.ones((1, 2), ctx=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('allreduce之前：\n', data[0], '\n', data[1])
allreduce(data)
print('allreduce之后：\n', data[0], '\n', data[1])

# 数据分发
# 我们需要一个简单的工具函数，将一个小批量数据均匀地分布在多个GPU上。 例如，有两个GPU时，我们希望每个GPU可以复制一半的数据
# data = np.arange(20).reshape(4, 5)
# devices = [npx.gpu(0), npx.gpu(1)]
# split = gluon.utils.split_and_load(data, devices)
# print('输入：', data)
# print('设备：', devices)
# print('输出：', split)

def split_batch(X, y, devices):
    """将X和y拆分到多个设备上"""
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, devices),
            gluon.utils.split_and_load(y, devices))

# 训练
# 计算图在小批量内的设备之间没有任何依赖关系，因此它是“自动地”并行执行
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    with autograd.record(): # 在每个GPU上分别计算损失
        ls = [loss(lenet(X_shard, device_W), y_shard)
              for X_shard, y_shard, device_W in zip(
                  X_shards, y_shards, device_params)]
    for l in ls: # 反向传播在每个GPU上分别执行
        l.backward()
    # 将每个GPU的所有梯度相加，并将其广播到所有GPU
    # 注意：每个gpu设备都拥有一套全量的权重参数, 比如权重w1在所有gpu都有一个副本
    # device_params是所有gpu权重的列表[gpu0_params(w1, w1 ...), gpu1_params(w1, w2 ...)]
    # 因此len(device_params[0])代表一个有多少个参数
    for i in range(len(device_params[0])):
        # 每个for循环将计算一个参数在所有gpu设备上的和然后广播到所有设备上
        # allreduce接收的是一个参数在所有设备上的列表
        allreduce([device_params[c][i].grad for c in range(len(devices))])

    # 在每个GPU上分别更新模型参数
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # 在这里，我们使用全尺寸的小批量

def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 将模型参数复制到num_gpus个GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
            timer.start()
            for X, y in train_iter:
                # 为单个小批量执行多GPU训练
                train_batch(X, y, device_params, devices, lr)
                npx.waitall()
            timer.stop()
             # 在GPU0上评估模型
            animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
                lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
          f'在{str(devices)}')

train(num_gpus=1, batch_size=256, lr=0.2)

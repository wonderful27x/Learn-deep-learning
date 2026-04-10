# 使用mxnet框架训练线性回归
from mxnet import autograd, gluon, np, npx
from d2l import mxnet as d2l
from mxnet.gluon import nn
from mxnet import init
from mxnet import gluon

npx.set_np()

# 生成数据
true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 读取数据到矩阵
def load_array(data_arrays, batch_size, is_train=True):
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 定义模型
net = nn.Sequential() # 分层的模型
net.add(nn.Dense(1))  # 添加一个一个输出的层

# 初始化模型参数
# 从均值为0，标准差为0.01的正态分布中随机采样来初始化权重，默认偏置为0
net.initialize(init.Normal(sigma=0.01))

# 定义损失函数, L2Loss均方误差
loss = gluon.loss.L2Loss()

# 定义优化算法
# net.collect_params获取需要优化的参数，sgd小批量梯度算法，超参数学习率
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# 训练
num_epochs = 3 # 迭代周期， 一个周期遍历完所有数据
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')

# 打印训练结果, [0]第0层
w = net[0].weight.data()
print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
b = net[0].bias.data()
print(f'b的估计误差：{true_b - b}')

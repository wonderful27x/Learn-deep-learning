from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
import time

npx.set_np()

# 注意力机制有三个概念：
# 1. 自注提示，也叫做查询
# 2. 非自注提示，也叫做键
# 3. 值，每个值和键都是配对的，给定一个查询，注意力机制通过注意力汇聚
# 将选择引导至感官输入，这个输入就是值
# 简单理解注意力权重：假设有一批训练数据集，分别有特征X，和标签Y，这里的X-Y则被看作键-值对，
# 然后有一个测试样本t_x, t_x就被看作一个查询，这个查询通过与键的关系如距离X-t_x, 可以得到一个注意力权重矩阵，如当t_x与训练样本的距离越小，分配的权重就越大，那么它对值Y的加权也就越大，就表示注意力更集中于该键对于的值，当距离为0时，权重为1

# 注意力汇聚：Nadaraya-Watson 核回归
# 生成数据
n_train = 50 # 训练样本
x_train = np.sort(np.random.rand(n_train)*5) # 排序后的训练样本

# 定义Y=f(x)
def f(x):
    return 2 * np.sin(x) + x**0.8

y_train = f(x_train) + np.random.normal(0.0, 0.5, (n_train,)) # 加上噪声的训练样本输出
x_test = np.arange(0, 5, 0.1) # 测试样本
y_truth = f(x_train) # 测试样本的真实输出
n_test = len(x_test)
print(n_test)

# 下面的函数将绘制所有的训练样本（样本由圆圈表示）， 不带噪声项的真实数据生成函数
# （标记为“Truth”）， 以及学习得到的预测函数（标记为“Pred”）。
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)

# 平均汇聚，先使用最简单的估计器来解决回归问题。 基于平均汇聚来计算所有训练样本输出值的平均值
y_hat = y_train.mean().repeat(n_test)
plot_kernel_reg(y_hat)

# 非参数注意力汇聚
# X_repeat的形状:(n_test,n_train),
# 每一行都包含着相同的测试输入（例如：同样的查询）
X_repeat = x_test.repeat(n_train).reshape((-1, n_train))
# x_train包含着键。attention_weights的形状：(n_test,n_train),
# 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
# 这里用的是高斯核得到的注意力汇聚算法, 具体需要10.2.3小节的公式推导
attention_weights = npx.softmax(-(X_repeat - x_train)**2 / 2)
y_hat = np.dot(attention_weights, y_train)
# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
plot_kernel_reg(y_hat)

# 绘制注意力权重热图
d2l.show_heatmaps(np.expand_dims(np.expand_dims(attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')

# 带参数的注意力汇聚，就是在x-xi的距离上乘以一个权重，这个权重是可以学习的

# test, 下面的测试是为了运算的理解
# 需要先理解批量矩阵乘法
X = np.arange(1, 6)
len_x = len(X)
X_tile = np.tile(X, (len_x, 1))
keys = X_tile[(1 - np.eye(len_x)).astype('bool')].reshape((len_x, -1))
queries_o = X.repeat(keys.shape[1])
queries = X.repeat(keys.shape[1]).reshape((-1, keys.shape[1]))
print(x_train)
print(X)
print(X_tile)
print(f"keys: {keys.shape}")
print(keys)
print(f"queries_o: {queries_o.shape}")
print(queries_o)
print(f"queries: {queries.shape}")
print(queries)
queries_keys_diff = queries - keys
attention_weights = npx.softmax(
            -((queries - keys) * 1)**2 / 2)
print(f"queries_keys_diff: {queries_keys_diff.shape}")
print(queries_keys_diff)
print(f"attention_weights: {attention_weights.shape}")
print(attention_weights)

# 定义模型
class NWKernelRegression(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = self.params.get('w', shape=(1,))

    # 这里必须理解qkv的输入是怎么处理的才能理解下面的运算
    # 总之就是在一个训练样本里，计算每个样本除了对自身以外关于其他所有key-value
    # 的权重，然后输出加权平均
    # 比如key是[1 2 3]那么需要计算1对[2 3] 2对[1 3] 3对[1 2]的权重，然后对对应key的
    # value进行加权，如1对[2 3]的权重矩阵是softmax([1-2, 1-3]), value是[10, 25]
    # 就根据对于权重对value进行加权得到输入1的预测
    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询数，“键－值”对数)
        queries = queries.repeat(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = npx.softmax(
                -((queries - keys) * self.w.data())**2 / 2)
        # values的形状为(查询数，“键－值”对数)
        return npx.batch_dot(np.expand_dims(self.attention_weights, 1),
                             np.expand_dims(values, -1)).reshape(-1)

# 训练
# 接下来，将训练数据集变换为键和值用于训练注意力模型。 在带参数的注意力汇聚模型中， 任何一个训练样本的输入都会和除自己以外的所有训练样本的“键－值”对进行计算， 从而得到其对应的预测输出。
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = np.tile(x_train, (n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = np.tile(y_train, (n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - np.eye(n_train)).astype('bool')].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - np.eye(n_train)).astype('bool')].reshape((n_train, -1))

net = NWKernelRegression()
net.initialize()
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
for epoch in range(5):
    with autograd.record():
        l = loss(net(x_train, keys, values), y_train)
    l.backward()
    trainer.step(1)
    print(f'epoch {epoch+1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
print(f'epoch {epoch+1}, loss {float(l.sum()):.6f}')

print(net.params.keys())
net_w = net.w.data()
print(f'weight: {net_w}')

# 预测
# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = np.tile(x_train, (n_test, 1))
# value的形状:(n_test，n_train)
values = np.tile(y_train, (n_test, 1))
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)

# 绘制热图
d2l.show_heatmaps(np.expand_dims(
                      np.expand_dims(net.attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')

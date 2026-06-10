import math
from mxnet import np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()

# 注意力评分函数，这一小节不再使用高斯核评分函数
# ***这里要理解最关键的是要搞懂注意力权重，它是每个query对所有key的打分***，然后再softmax
# 如果有m个query，k个key，那么最终要计算得到的矩阵就是mxk的形状
# 每一行都是该query对所有key的打分, 至于要用什么手段来打分，这就取决于打分函数，
# 如前一节通过q-k的距离来打分

# 掩蔽softmax
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = valid_lens.repeat(shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, True, value=-1e6, axis=1)
        return npx.softmax(X).reshape(shape)

# 加性注意力
# 看10.3.2小节(https://zh.d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html)
# 将查询和键连结起来后输入到一个多层感知机（MLP）中， 感知机包含一个隐藏层，其隐藏单元数是一个超参数h
# 其中查询和键的学习参数分别是权重W(hxq)和W(hxk), q是查询的数量，k是键的数量
# 还有一个W(hx1)，a(q,k) = W(hx1)tanh(Wq + Wk)
# 也就是查询和键都是一个全连接隐藏层，隐藏单元为h, 然后将它们相加
class AdditiveAttention(nn.Block):
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # 使用'flatten=False'只转换最后一个轴，以便其他轴的形状保持不变
        self.W_k = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_q = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.w_v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # 这里要理解的是前面flatten=False只对最后一个轴做转换
        # 最后一个轴对应的就是query或key的特征维度d
        # 如queries(batch_size, num_q, d_q) * W(d_q, num_hidden)-> queries(batch_size, num_q, num_hidden)
        # 如   keys(batch_size, num_k, d_k) * W(d_k, num_hidden)-> queries(batch_size, num_k, num_hidden)
        # 它们都是对最后一个轴d进行转换，都变成了相同的num_hidden
        # 因此即使输入的queries和keys的特征不一样，它们后面也能进行相加
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播的方式进行求和
        # 这里要做的就是得到每个查询query对所有key的打分矩阵，知道这个就可以了
        # 最终的：(batch_size，查询的个数，“键－值”对的个数，num_hidden)
        # (查询个数，“键－值”对的个数)正是我们想要的每个查询对所有键的打分矩阵，
        # num_hiddens是查询或键的特征表示的维度，后面要把他变成一维变成一个分数，也就是self.w_w = nn.Dense(1...)的作用
        features = np.expand_dims(queries, axis=2) + np.expand_dims(keys, axis=1)
        # 激活函数
        features = np.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = np.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        # 根据权重矩阵对值进行加权
        # 输出的形状: (batch_size, 查询的个数, 值的维度)
        # 输入(全连接层转换后)queries的形状：(batch_size，查询的个数，num_hidden)
        # 输入(全连接层转换后)key的形状：(batch_size，“键－值”对的个数，num_hiddens)
        # 这里由于有一个中间转换，看下面的缩放点击注意力更直观
        return npx.batch_dot(self.dropout(self.attention_weights), values)

# test
# 其中查询、键和值的形状为（批量大小，步数或词元序列长度，特征大小）
# 前面说了因为隐藏单元一样, queries和keys的特征维度可以不一样
queries, keys = np.random.normal(0, 1, (2, 1, 20)), np.ones((2, 10, 2))
# values的小批量数据集中，两个值矩阵是相同的
values = np.arange(40).reshape(1, 10, 4).repeat(2, axis=0)
valid_lens = np.array([2, 6])
attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.initialize()
attention_value = attention(queries, keys, values, valid_lens)
print("加性注意力")
print("W_q.weight: ", attention.W_q.weight.shape)
print("W_k.weight: ", attention.W_k.weight.shape)
print("w_v.weight: ", attention.w_v.weight.shape)
print("attention_value shape: ", attention_value.shape)
print(attention_value)
print("attention_weights shape:\n", attention.attention_weights.shape)
print(attention.attention_weights)
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')


# 缩放点积注意力
# 用查询和键的点积乘以d的平方根作为评分函数，这要求查询和键的特征维度d一样
class DotProductAttention(nn.Block):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状: (batch_size，)或者(batch_size，查询的个数)
    # 最终输出形状: (batch_size, 查询个数, 值的维度)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = npx.batch_dot(queries, keys, transpose_b=True) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)

        return npx.batch_dot(self.dropout(self.attention_weights), values)

queries = np.random.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.initialize()
attention_weights = attention(queries, keys, values, valid_lens)
print("缩放点积注意力")
print(attention_value)
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')


# 10.5. 多头注意力
# 多头注意力是先将查询、键和值进行线性投影，映射到一个新的空间产生h组值，然后将这h组
# 变换后的查询、键和值并行送到注意力汇聚，最后将它们的输出拼接在一起。这样每一组注意力汇聚
# (称为头head)可以得到不同类别的注意焦点的注意力
# 比如: “猫坐在垫子上，因为它很柔软。”
# 普通注意力会：
# 计算“它”与句中每个词的相关性
# 给“垫子”最高权重（因为“它”指代垫子）
# 给“柔软”较高权重（因为描述垫子）
# 给“猫”较低权重
# 4个头可能分别关注：
# 头编号	关注的关系类型	对“它”的注意力分布
# Head 1	指代关系	“垫子”权重最高
# Head 2	属性关系	“柔软”权重最高
# Head 3	空间关系	“坐在”权重较高
# Head 4	对比关系	“猫”（对比）权重较高

# 多头注意力每头head的隐藏单元数为原num_hiddens/heads
# 多头注意力融合了来自于多个注意力汇聚的不同知识，这些知识的不同来源于相同的查询、键和值的不同的子空间表示。
class MultiHeadAttention(nn.Block):
    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        # 使用缩放点积注意力
        self.attention = DotProductAttention(dropout)
        # 查询、键、值的线性映射
        self.W_q = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        # 最终的拼接, 拼接后输出的形状和单头的形状一样的
        self.W_o = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = valid_lens.repeat(self.num_heads, axis=0)

        # 注意这里经过转换后输入attention, 相当于并行运算多个head
        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        self.attention_weights = self.attention.attention_weights

        # output_concat的形状:(batch_size，查询的个数，num_hiddens), 和单头完全一致
        output_concat = transpose_output(output, self.num_heads)
        # 需要注意的是W_o变换并未改变输出的形状，然而它是必要的
        # 在上面只是简单地将每个头学到的东西拼接，这里进行线性组合, 是真正的融合
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.transpose(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

# test 多头注意力输出的形状是（batch_size，num_queries，num_hiddens）。
print("多头注意力")
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()

batch_size, num_queries = 2, 4
num_kvpairs, valid_lens = 6, np.array([3, 2])
X = np.ones((batch_size, num_queries, num_hiddens))
Y = np.ones((batch_size, num_kvpairs, num_hiddens))
print(attention(X, Y, Y, valid_lens).shape)

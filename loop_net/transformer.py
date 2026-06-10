import os
import collections
import math
import pandas as pd
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn, rnn

npx.set_np()

# 这一节是将注意机制加入到机器翻译中来
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

print("测试masked_softmax")
X = np.ones((2, 3, 4))
L = np.array([[1, 1, 1],
              [2, 2, 2]])
soft_x = masked_softmax(X, L)
print(soft_x)

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
        # !!!由于我们要对scores进行softmax操作，因此这里的valid_lens应该对应的是key-value的有效个数, 后面我们将看到它是编码器输出有效时间步数量
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        # 根据权重矩阵对值进行加权
        return npx.batch_dot(self.dropout(self.attention_weights), values)

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

# 简单的注意力是没有位置关系的，前面我们将注意力引入到rnn循环神经网络中，
# 是rnn内部建立的顺序，这里讨论的位置编码是利用正弦余弦波，将位置信息加再输入中
# 同一个频率的波表示不同的位置，不同频率的波则区分特征维度
# 具体公式查看10.6.3小节
class PositionalEncoding(nn.Block):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)




d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
              encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])

# 我们用空格代替不间断空格（non-breaking space）， 使用小写字母替换大写字母，并在单词和标点符号之间插入空格
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])

# 词元化，每个词元是一个单词或标点，
# 此函数返回两个词元列表：source和target： source[i]是源语言（这里是英语）第i
# 个文本序列的词元列表， target[i]是目标语言（这里是法语）第i
# 个文本序列的词元列表。
def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
print(source[:6])
print(target[:6])

# 绘制直方图，统计每个文本序列的词元数量
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target);

# 词表
# 对词进行频率统计，并分配索引，将词转换为数字
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return[self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# 词表
src_vocab = Vocab(source, min_freq=2,
                  reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)

# 加载数据集
# num_steps指定时间步数或词元数量，由于每个文本序列的单词数不一样，因此小于num_steps则填充<pad>，否则截断
def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))

# 生成小批量训练数据集，每个序列加上结束特定符号<eos>
def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = np.array([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).astype(np.int32).sum(1)
    return array, valid_len

# 返回数据迭代器
def load_data_nmt(batch_size, num_steps, num_examples=600):
     """返回翻译数据集的迭代器和词表"""
     text = preprocess_nmt(read_data_nmt())
     source, target = tokenize_nmt(text, num_examples)
     src_vocab = Vocab(source, min_freq=2,
                       reserved_tokens=['<pad>', '<bos>', '<eos>'])
     tgt_vocab = Vocab(target, min_freq=2,
                       reserved_tokens=['<pad>', '<bos>', '<eos>'])
     src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
     tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
     data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
     data_iter = d2l.load_array(data_arrays, batch_size)
     return data_iter, src_vocab, tgt_vocab

# 打印函数
def print_trainer_data(data, src_vocab, tgt_vocab):
    print("小批量数据集测试:")
    X, X_valid_len, Y, Y_valid_len = data
    print('X: ', X.astype(np.int32))
    print('X的有效长度: ', X_valid_len)
    print('Y: ', Y.astype(np.int32))
    print('Y的有效长度: ', Y_valid_len)

    for i in range(len(X_valid_len)):
        row = X.astype(np.int32)[i]
        print("x-idx: ", row)
        print("x-str: ", src_vocab.to_tokens(row.tolist()), " valid_len:", X_valid_len[i])

        row = Y.astype(np.int32)[i]
        print("y-idx: ", row)
        print("y-str: ", tgt_vocab.to_tokens(row.tolist()), " valid_len:", Y_valid_len[i])

# 读取第一个小批量数据
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for data in train_iter:
    print_trainer_data(data, src_vocab, tgt_vocab)
    break

# 编解码器架构
# “编码器－解码器”架构可以将长度可变的序列作为输入和输出，因此适用于机器翻译等序列转换问题。

# 编码器将长度可变的序列作为输入，并将其转换为具有固定形状的编码状态。

# 解码器将具有固定形状的编码状态映射为长度可变的序列。

class Encoder(nn.Block):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Block):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Block):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

# 注意力解码器接口
class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


# Transformer作为编码器－解码器架构的一个实例，与基于Bahdanau注意力实现的序列到序列的学习相比，Transformer的编码器和解码器是基于自注意力的模块叠加而成的，源（输入）序列和目标（输出）序列的嵌入（embedding）表示将加上位置编码（positional encoding），再分别输入到编码器和解码器中。
# Transformer同时使用了多头自注意力（multi-head self-attention）汇聚、基于位置的前馈网络（positionwise feed-forward network）、残差网络和层规范化等技术，知识较广
# 具体参见10.7.1. 模型 https://zh-v2.d2l.ai/chapter_attention-mechanisms/transformer.html#id3

# 基于位置的前馈网络
class PositionWiseFFN(nn.Block):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Dense(ffn_num_hiddens, flatten=False,
                               activation='relu')
        self.dense2 = nn.Dense(ffn_num_outputs, flatten=False)

    def forward(self, X):
        return self.dense2(self.dense1(X))

print("基于位置的前馈网络测试")
ffn = PositionWiseFFN(4, 8)
ffn.initialize()
print(ffn(np.ones((2, 3, 4)))[0])

# 残差链接和层规范化
# 层规范化是对一个样本的特征维度进行规范化，批量规范化则是跨样本对不同批次相同位置特征的规范化
print("层规范化和批量规范化对比测试")
ln = nn.LayerNorm()
ln.initialize()
bn = nn.BatchNorm()
bn.initialize()
X = np.array([[1, 2], [2, 3]])
# 在训练模式下计算X的均值和方差
with autograd.record():
    print('层规范化:', ln(X), '\n批量规范化: ', bn(X))

class AddNorm(nn.Block):
    """残差连接后进行层规范化"""
    def __init__(self, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm()

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

# 编码器
# EncoderBlock类包含两个子层：多头自注意力和基于位置的前馈网络，
# 这两个子层都使用了残差连接和紧随的层规范化。
class EncoderBlock(nn.Block):
    """Transformer编码器块"""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        # 输出是形状：(batch_size, 查询个数, num_hiddens)
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        # 残差链接层规范化不会改下形状
        self.addnorm1 = AddNorm(dropout)
        # 可以看到FFN进行了扩维ffn_num_hiddens, 然后非线性激活，最后又缩会num_hiddens维
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

print("transformer编码器层测试")
X = np.ones((2, 100, 24))
valid_lens = np.array([3, 2])
encoder_blk = EncoderBlock(24, 48, 8, 0.5)
encoder_blk.initialize()
# 因为num_hiddens和输入X的特征维度一样，所以编码层不会改变输入形状
print(encoder_blk(X, valid_lens).shape)

# 下面实现的Transformer编码器的代码中，堆叠了num_layers个EncoderBlock类的实例。由于这里使用的是值范围在-1和1之间的固定位置编码，因此通过学习得到的输入的嵌入表示的值需要先乘以嵌入维度的平方根进行重新缩放，然后再与位置编码相加。
class TransformerEncoder(Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(
                    # 使用和embedding相同的num_hiddens，输出形状和输入一致
                    EncoderBlock(num_hiddens, ffn_num_hiddens, num_heads,
                                 dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

print("TransformerEncoder测试")
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
encoder.initialize()
# Transformer编码器输出的形状是（批量大小，时间步数目，num_hiddens）
print(encoder(np.ones((2, 100)), valid_lens).shape)

# 解码器
# Transformer解码器也是由多个相同的层组成。在DecoderBlock类中实现的每个层包含了三个子层：解码器自注意力、“编码器-解码器”注意力和基于位置的前馈网络。这些子层也都被残差连接和紧随的层规范化围绕。
# 在掩蔽多头解码器自注意力层（第一个子层）中，查询、键和值都来自上一个解码器层的输出。关于序列到序列模型（sequence-to-sequence model），在训练阶段，其输出序列的所有位置（时间步）的词元都是已知的；然而，在预测阶段，其输出序列的词元是逐个生成的。因此，在任何解码器时间步中，只有生成的词元才能用于解码器的自注意力计算中。为了在解码器中保留自回归的属性，其掩蔽自注意力设定了参数dec_valid_lens，以便任何查询都只会与解码器中所有已经生成词元的位置（即直到该查询位置为止）进行注意力计算。
# 注意由于transformer中没有像之前的rnn网络那样按照时间步进行计算的过程，训练时输入的X的所有时间步是并行的，所以需要掩蔽操作，如下：
#   a b c   有效长度
# a o x x    1
# b o o x    2
# c o o o    3
# 也就是a只能看到自己，b不能看到它前面的a和自己
# 这样x的看不见的就设置权重为0
class DecoderBlock(nn.Block):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        # 这是解码器自注意力, 它的查询、键和值都来自上一个解码器层的输出
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(dropout)
        # “编码器-解码器”注意力, 它的查询来自前一个解码器层的输出，而键和值来自整个编码器的输出
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = np.concatenate((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        # 处理掩蔽
        if autograd.is_training():
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            # 这样处理后在注意力层里面的masked_softmax就会屏蔽掉看不见的元素
            # 假如时间步是3，那么[1, 2, 3]代表第1，2，3项查询的键的长度分别为1，2，3
            # 可视化入下
            #   a b c   有效长度
            # a o x x    1
            # b o o x    2
            # c o o o    3
            dec_valid_lens = np.tile(np.arange(1, num_steps + 1, ctx=X.ctx),
                                     (batch_size, 1))
        else:
            # 预测时是按时间步输入的，不需要掩蔽
            dec_valid_lens = None

        # 解码器自注意力, 带掩码
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # “编码器－解码器”注意力。
        # 'enc_outputs'的开头:('batch_size','num_steps','num_hiddens')
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

print("Transformer解码器层测试")
decoder_blk = DecoderBlock(24, 48, 8, 0.5, 0)
decoder_blk.initialize()
X = np.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
print(decoder_blk(X, state)[0].shape)

# 现在我们构建了由num_layers个DecoderBlock实例组成的完整的Transformer解码器。最后，通过一个全连接层计算所有vocab_size个可能的输出词元的预测值。解码器的自注意力权重和编码器解码器注意力权重都被存储下来，方便日后可视化的需要。
class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add(
                    DecoderBlock(num_hiddens, ffn_num_hiddens, num_heads,
                                 dropout, i))
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # 编码器－解码器自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


# 损失函数
# 因为填充的词源损失计算需要屏蔽，所以特殊处理
# 我们可以通过扩展softmax交叉熵损失函数来遮蔽不相关的预测。 最初，所有预测词元的掩码都设置为1。 一旦给定了有效长度，与填充词元对应的掩码将被设置为0。 最后，将所有词元的损失乘以掩码，以过滤掉损失中填充词元产生的不相关预测
class MaskedSoftmaxCELoss(gluon.loss.SoftmaxCELoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = np.expand_dims(np.ones_like(label), axis=-1)
        weights = npx.sequence_mask(weights, valid_len, True, axis=1)
        # print(weights)
        return super(MaskedSoftmaxCELoss, self).forward(pred, label, weights)

# test
print("测试loss...")
loss = MaskedSoftmaxCELoss()
loss(np.ones((3, 4, 10)), np.ones((3, 4)), np.array([4, 2, 0]))


# 训练
# 在下面的循环训练过程中，如 图9.7.1所示， 特定的序列开始词元（“<bos>”）和 原始的输出序列（不包括序列结束词元“<eos>”） 拼接在一起作为解码器的输入。 这被称为强制教学（teacher forcing）， 因为原始的输出序列（词元的标签）被送入解码器。 或者，将来自上一个时间步的预测得到的词元作为解码器的当前输入。

def train_seq2seq(net, data_iter, lr, num_epochs, src_vocab, tgt_vocab, device):
    net.initialize(init.Xavier(), force_reinit=True, ctx=device)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    loss = MaskedSoftmaxCELoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)
        for batch in data_iter:
            # print_trainer_data(batch, src_vocab, tgt_vocab)
            X, X_valid_len, Y, Y_valid_len = [
                    x.as_in_ctx(device) for x in batch]
            bos = np.array([tgt_vocab['<bos>']] * Y.shape[0],
                           ctx=device).reshape(-1, 1)
            # 这里就是标签作为解码器的输入
            # 这里需要理解添加eos bos pad和训练步骤的对应
            # 首先X和Y都在数据后面加上了结束符号<eos>，然后通过加<pad>或截断统一时间步，
            # 假如时间步都统一到6, 如下：
            # X = a b c eos pad pad
            # Y = 1 2 3 eos pad pad
            # 构造解码器输入dec_input，bos开头，去掉最后一个元素
            # dec_input = bos 1 2 3 eos pad
            # 训练步骤如下:
            # 时间步1: enc(X) + bos -> 1
            # 时间步2: enc(X) +   1 -> 2
            # 时间步3: enc(X) +   2 -> 3
            # 时间步4: enc(X) +   3 -> eos
            # 时间步5: enc(X) + eos -> pad
            # 时间步6: enc(X) + pad -> pad
            # 也就是说：
            # 输入: X = a b c eos pad pad
            # 预测: Y = 1 2 3 eos pad pad
            dec_input = np.concatenate([bos, Y[:, :-1]], 1)
            with autograd.record():
                Y_hat, _ = net(X, dec_input, X_valid_len)
                l = loss(Y_hat, Y, Y_valid_len)
            l.backward()
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            trainer.step(num_tokens)
            metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1], ))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f}'
          f'tokens/sec on {str(device)}')

num_hiddens, num_layers, dropout = 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_hiddens, num_heads = 64, 4

print("训练...")
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
encoder = TransformerEncoder(
        len(src_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout)
decoder = TransformerDecoder(
        len(tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout)
net = EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, src_vocab, tgt_vocab, device)
# d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

# 预测
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weight=False):
    """序列到序列模型的预测"""
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_lens = np.array([len(src_tokens)], ctx=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = np.expand_dims(np.array(src_tokens, ctx=device), axis=0)
    # print(f'encoder input shape: {enc_X.shape}')
    enc_outputs = net.encoder(enc_X, enc_valid_lens)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_lens)
    # 添加批量轴
    dec_X = np.expand_dims(np.array([tgt_vocab['<bos>']], ctx=device), axis=0)
    output_seq, attension_weight_seq = [], []
    # 注意这里是时间步的for循环，也就是说一个词元一个词元地预测
    # 因为但我们预测下一个词元时必须知道前面预测得到的词元
    for _ in range(num_steps):
        # print(f'decoder input shape: {dec_X.shape}')
        Y, dec_state = net.decoder(dec_X, dec_state)
        # print(f'decoder 输出Y: {Y}')
        # (batch_size,num_steps,vocab_size)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(axis=2)
        pred = dec_X.squeeze(axis=0).astype('int32').item()
        # 保存注意力权重
        if save_attention_weight:
            attension_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attension_weight_seq

# 预测序列的评估
def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

print("预测...")
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = predict_seq2seq(
    # translation, dec_attention_weight_seq = d2l.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

# 当进行最后一个英语到法语的句子翻译工作时，让我们可视化Transformer的注意力权重。编码器自注意力权重的形状为（编码器层数，注意力头数，num_steps或查询的数目，num_steps或“键－值”对的数目）。
enc_attention_weights = np.concatenate(net.encoder.attention_weights, 0).reshape((num_layers,
    num_heads, -1, num_steps))
print(enc_attention_weights.shape)
d2l.show_heatmaps(
    enc_attention_weights, xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

# 为了可视化解码器的自注意力权重和“编码器－解码器”的注意力权重，我们需要完成更多的数据操作工作。例如用零填充被掩蔽住的注意力权重。值得注意的是，解码器的自注意力权重和“编码器－解码器”的注意力权重都有相同的查询：即以序列开始词元（beginning-of-sequence,BOS）打头，再与后续输出的词元共同组成序列。
dec_attention_weights_2d = [np.array(head[0]).tolist()
                            for step in dec_attention_weight_seq
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = np.array(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.transpose(1, 2, 3, 0, 4)
print(dec_self_attention_weights.shape, dec_inter_attention_weights.shape)
# Plusonetoincludethebeginning-of-sequencetoken
d2l.show_heatmaps(
    dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
    xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

d2l.show_heatmaps(
    dec_inter_attention_weights, xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))

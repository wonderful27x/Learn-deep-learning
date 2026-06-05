import os
import collections
import math
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

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Block):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_valid_lens, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, enc_valid_lens, *args)
        return self.decoder(dec_X, dec_state)


# 序列到序列学习的机器翻译实现
# 编码器
# 编码器将长度可变的输入序列转换成 形状固定的上下文变量
class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层, 可学习的权重表示，和独热编码不同，但作用一样
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, **args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.swapaxes(0, 1)
        state = self.rnn.begin_state(batch_size=X.shape[1], ctx=X.ctx)
        output, state = self.rnn(X, state)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state

# test
print("测试encoder...")
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.initialize()
X = np.zeros((4, 7))
output, state = encoder(X)
print(output.shape)
print(len(state))
print(state[0].shape)

# 解码器
class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, enc_valid_lens, **args):
        return enc_outputs[1]
        # outputs的形状为(num_steps，batch_size，num_hiddens)
        # hidden_state[0]的形状为(num_layers，batch_size，num_hiddens)

    def forward(self, X, state):
        # print(f'decoder input shape: {X.shape}')
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).swapaxes(0, 1)
        # print(f'decoder input shape: {X.shape}')
        # 取最后一层的隐藏状态 -> context的形状:(batch_size,num_hiddens)
        context = state[0][-1]
        # 广播context，使其具有与X相同的num_steps,
        # 将编码器的输出state和X合并作为解码器输入，
        # 注意这里是将编码器最后一个时间步的state拼接到解码器的每一个时间步上, 相当于做了多份拷贝
        # 这里我们应该联想到因为拼接，所以权重矩阵W_x也被扩大了，
        # 因为W_x的大小为inputsxnum_hiddens = (embed_size + num_hiddens) x num_hiddens = (32+32) x 32
        # 这里可以测试一下, !!!需要用RNN网络测试, GRU需要x3
        # print("解码器网络参数keys：", self.rnn.collect_params().keys)
        # print("解码器网络权重矩阵：", self.rnn.collect_params())
        # 这里特别需要理解的是,X实际是标签（label(t-1)),
        # 将标签(不包含最后一个)作为输入,这称为强制教学，
        # 按道理应该是用前面预测的输出作为输入去预测下一个词，
        # 这里使用的确实标签，这样做有很多好处, 比如避免误差累计，收敛快等, 总之是通用做法
        # 这里需要理解，内部算法有一个按时间步的for循环的计算过程就可以了
        # 也就是按时间步进行输入，然后得到下一个预测词源
        context = np.broadcast_to(context, (
            X.shape[0], context.shape[0], context.shape[1]))
        X_and_context = np.concatenate((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).swapaxes(0, 1)
        # print("X.shape: ", X.shape, "X_and_context.shape: ", X_and_context.shape)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state

# test decoder
print("测试decoder...")
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
decoder.initialize()
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
print(output.shape)
print(len(state))
print(state[0].shape)

# 注意力解码器接口
class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(num_steps，batch_size，num_hiddens)
        # hidden_state[0]的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state[0]的形状为(num_layers,batch_size,
        # num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        # 这里自己循环每个时间步是因为每个时间步输入的注意力不一样
        # 外层进行循环rnn内部能根据输入的时间步数识别出来
        for x in X:
            # query的形状为(batch_size,1,num_hiddens)
            query = np.expand_dims(hidden_state[0][-1], axis=1)
            # context的形状为(batch_size,1,num_hiddens)
            # query查询是解码器上一时间步最后一层隐状态
            # 键-值对是编码器最后一层所有时间步隐状态
            # 因此这里的context上下文就是编码器隐状态按查询对时间步的一个加权平均，
            # 不再是之前的统统用最后一个时间步的隐状态
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上连结
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens), 因为手动for循环只有一个时间步了
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)

        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state, enc_valid_lens]

    def attention_weights(self):
        return self._attention_weights

# 测试
print("注意力解码器测试...")
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.initialize()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
decoder.initialize()
X = np.zeros((4, 7)) # (batch_size, num_steps)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
print(output.shape)
print(len(state))
print(state[0].shape)
print(len(state[1]))
print(state[1][0].shape)


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
                # Y_hat, _ = net(X, dec_input)
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

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

print("训练...")
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
# decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, src_vocab, tgt_vocab, device)

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
    enc_outputs = net.encoder(enc_X)
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
            attension_weight_seq.append(net.decoder.attention_weights())
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
    has_attention_weight = True if isinstance(decoder, AttentionDecoder) else False
    translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, has_attention_weight)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

attention_weights = np.concatenate([step[0][0][0] for step in attention_weight_seq], 0
    ).reshape((1, 1, -1, num_steps))
# 加上一个包含序列结束词元
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1],
    xlabel='Key positions', ylabel='Query positions')

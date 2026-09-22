import os
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()


d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')

def read_voc_images(voc_dir, is_train=True, limit=0):
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        if limit > 0 and i >= limit:
            break
        features.append(image.imread(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(image.imread(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png')))
    return features, labels

# train_features, train_labels = read_voc_images(voc_dir, True)
# n = 5
# imgs = train_features[0:n] + train_labels[0:n]
# d2l.show_images(imgs, 2, n);

#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

#@save
def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

# # 飞机头部区域的类别索引为1，而背景索引为0。
# y = voc_label_indices(train_labels[0], voc_colormap2label())
# print(y[105:115, 130:140], VOC_CLASSES[1])

#数据预处理，统一尺寸，缩放对于语义分割不够精确，使用随机裁剪
def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label

# imgs = []
# for _ in range(n):
#     imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
# d2l.show_images(imgs[::2] + imgs[1::2], 2, n);

# 通过继承高级API提供的Dataset类，自定义了一个语义分割数据集类VOCSegDataset。 通过实现__getitem__函数，我们可以任意访问数据集中索引为idx的输入图像及其每个像素的类别索引。 由于数据集中有些图像的尺寸可能小于随机裁剪所指定的输出尺寸，这些样本可以通过自定义的filter函数移除掉。 此外，我们还定义了normalize_image函数，从而对输入图像的RGB三个通道的值分别做标准化。
#@save
class VOCSegDataset(gluon.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""
    def __init__(self, is_train, crop_size, voc_dir, limit=0):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train, limit=limit)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature.transpose(2, 0, 1),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

# 加载数据，返回迭代器
def load_data_voc(batch_size, crop_size, limit=0):
    """加载VOC语义分割数据集"""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = gluon.data.DataLoader(
            VOCSegDataset(True, crop_size, voc_dir, limit), batch_size,
            shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gluon.data.DataLoader(
            VOCSegDataset(False, crop_size, voc_dir, limit), batch_size,
            last_batch='discard', num_workers=num_workers)
    return train_iter, test_iter


# batch_size = 64
# crop_size = (320, 480)
# train_iter, test_iter = load_data_voc(batch_size, crop_size)


# ========构造模型, 使用ResNet-18预训练模型=======================
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
# 打印模型输出层，和倒数前三层（除去输出层）
print(pretrained_net.features[-3:], pretrained_net.output)
# 通过上面的打印可以看出，模型的features最后两层是
# Flatten（展开为一维）和GlobalAvgPool2D（全局平均汇聚层）
# 这两层不需要，去掉
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)

# 打印形状
X = np.random.uniform(size=(1, 3, 320, 480))
print(net(X).shape)

# 接下来需要
# 1. 增加一个1x1卷积层将输出通道转换为我们的数据集中的类别（21）类
# 2. 使用转置卷积将特征图变换会输入图的宽高，因为我们是做像素级别的分类，
#    即每个像素属于那种类别，关于转置卷积核核padding,strides的计算，
#    和普通卷积参数一样, 需要参考上面的打印和6.3节的计算方法
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16, strides=32))

# 初始化转置卷积层，使用双线性插值，
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)

# 我们打印下图像，可以看到转置卷积实现的双线性插值的放大效果
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
img = image.imread('../../../videoimg/yunyun.jpeg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());

# 初始化我们加入的两层模型
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())

# 读取数据集
batch_size, crop_size = 32, (320, 480)
# limit = batch_size * 10
limit = 0
train_iter, test_iter = load_data_voc(batch_size, crop_size, limit)

# 训练
num_epochs, lr, wd = 5, 0.1, 1e-3
devices = d2l.try_all_gpus()
devices = [d2l.try_gpu(0)]
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
print(devices)
print(len(train_iter))
print(len(test_iter))
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


# 预测
# 在预测时，我们需要将输入图像在各个通道做标准化，并转成卷积神经网络所需要的四维输入格式。
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    # (H W C) -> (N C H W)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    # 输出shape是（N C H W），每个像素有C个类别，
    # argmax(axis=1)取评分最高的 -> (N H W) reshape() -> (H W)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

# 为了可视化预测的类别给每个像素，我们将预测类别映射回它们在数据集中的标注颜色。
def label2image(pred):
    colormap = np.array(VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]

# 测试数据集中的图像大小和形状各异。 由于模型使用了步幅为32的转置卷积层，因此当输入图像的高或宽无法被32整除时，转置卷积层输出的高或宽会与输入图像的尺寸有偏差。 为了解决这个问题，我们可以在图像中截取多块高和宽为32的整数倍的矩形区域，并分别对这些区域中的像素做前向传播。 请注意，这些区域的并集需要完整覆盖输入图像。 当一个像素被多个区域所覆盖时，它在不同区域前向传播中转置卷积层输出的平均值可以作为softmax运算的输入，从而预测类别。

# 为简单起见，我们只读取几张较大的测试图像，并从图像的左上角开始截取形状为320x480
# 的区域用于预测。 对于这些测试图像，我们逐一打印它们截取的区域，再打印预测结果，最后打印标注的类别。
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = read_voc_images(voc_dir, False, limit)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);

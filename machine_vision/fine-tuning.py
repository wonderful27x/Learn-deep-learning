# %matplotlib inline
import os
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()


# 本节将介绍迁移学习中的常见技巧:微调（fine-tuning）。如 图13.2.1所示，微调包括以下四个步骤。
# 1. 在源数据集（例如ImageNet数据集）上预训练神经网络模型，即源模型。
# 2. 创建一个新的神经网络模型，即目标模型。这将复制源模型上的所有模型设计及其参数（输出层除外）。我们假定这些模型参数包含从源数据集中学到的知识，这些知识也将适用于目标数据集。我们还假设源模型的输出层与源数据集的标签密切相关；因此不在目标模型中使用该层。
# 3. 向目标模型添加输出层，其输出数是目标数据集中的类别数。然后随机初始化该层的模型参数。
# 4. 在目标数据集（如椅子数据集）上训练目标模型。输出层将从头开始进行训练，而所有其他层的参数将根据源模型的参数进行微调。

d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')

train_imgs = gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);

# 在训练期间，我们首先从图像中裁切随机大小和随机长宽比的区域，然后将该区域缩放为224x224
# 输入图像。 在测试过程中，我们将图像的高度和宽度都缩放到256像素，然后裁剪中央224x224
# 区域作为输入。 此外，对于RGB（红、绿和蓝）颜色通道，我们分别标准化每个通道。 具体而言，该通道的每个值减去该通道的平均值，然后将结果除以该通道的标准差。

# 使用RGB通道的均值和标准差，以标准化每个通道
# 注意这里的数值是ImageNet预训练模型数据集的均值和标准差
normalize = gluon.data.vision.transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop(224),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    normalize])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    normalize])

# 定义目标模型
# 下载预训练模型
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
# 预训练的源模型实例包含两个成员变量：features和output。 前者包含除输出层以外的模型的所有层，后者是模型的输出层。 此划分的主要目的是促进对除输出层以外所有层的模型参数进行微调。 源模型的成员变量output如下所示。
print(pretrained_net.output)

# 目标模型除了最后输出层换成自己的(2)和原型一样，然后使用原模型的参数初始化目标模型
# 这里我们用来识别热狗，只有是和不是两类
finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
# 输出层中的学习率比其他层的学习率大十倍
finetune_net.output.collect_params().setattr('lr_mult', 10)

# 微调模型
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gluon.data.DataLoader(
            train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
            test_imgs.transform_first(test_augs), batch_size)
    # devices = d2l.try_all_gpus()
    devices = [d2l.try_gpu(0)]
    net.collect_params().reset_ctx(devices)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# 微调预训练模型
print("微调训练...")
train_fine_tuning(finetune_net, 0.01)

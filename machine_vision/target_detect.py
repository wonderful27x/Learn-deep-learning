import os
import pandas as pd
from mxnet import gluon, image, np, npx
from d2l import mxnet as d2l

npx.set_np()

# 该数据集中每张图片里面有一个香蕉
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

# 读取数据集
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        # target的数据是这样的：
        # label      0
        # xmin     104
        # ymin      20
        # xmax     143
        # ymax      58
        # Name: 0.png, dtype: int64
        # list(target)获取到值: [0, 104, 20, 143, 58]
        targets.append(list(target))
        # print(img_name)
        # print(target.shape)
        # print(target)
        # print(list(target))
        # print(targets[0])
        # print(np.array(targets).shape)
        # print(np.expand_dims(np.array(targets), 1).shape)
        # np.array(targets)转为矩阵形式，shape: (targets长度, 5)
        # np.expand_dims(np.array(targets), 1)扩展维度1，shape: (targets长度, 1, 5)
        # 因此targets[0]代表的就是images[0]对应的目标边界框的数据
        # (1, 5)代表1个边界框，5代表: 类别，左上角x，左上角y，右下角x，右下角y
    return images, np.expand_dims(np.array(targets), 1) / 256


class BananasDataset(gluon.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        # transpose将图像HWC转为CHW
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)


# 批量处理最终得到
# 图像的小批量的形状为（批量大小、通道数、高度、宽度）
# 标签的小批量的形状为（批量大小，m，5）, 这里m代表图像中边界框的数量，这里是1
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter


batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(len(batch))
print(batch[0].shape, batch[1].shape)

# show
imgs = (batch[0][0:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])

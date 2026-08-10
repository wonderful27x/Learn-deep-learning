from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
import os

npx.set_np()

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32, ctx=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_ious >= iou_threshold)[0]
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * np.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = np.concatenate([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.ctx, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = np.tile((np.expand_dims((anchors_bbox_map >= 0),
                                            axis=-1)), (1, 4)).astype('int32')
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = np.zeros(num_anchors, dtype=np.int32, ctx=device)
        assigned_bb = np.zeros((num_anchors, 4), dtype=np.float32,
                                ctx=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = np.nonzero(anchors_bbox_map >= 0)[0]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].astype('int32') + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = np.stack(batch_offset)
    bbox_mask = np.stack(batch_mask)
    class_labels = np.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

# 单发多框检测SSD，

# 13.7.1.1. 类别预测层
# 类别预测层使用一个保持输入高和宽的卷积层。 这样一来，输出和输入在特征图宽和高上的空间坐标一一对应。 考虑输出和输入同一空间坐标(x, y):
# 输出特征图上（x, y）坐标的通道里包含了以输入特征图(x, y)坐标为中心生成的所有锚框的类别预测。 因此输出通道数为a(q+1), 其中索引为i(q+1) + j(0 <= j <= q)的通道代表了索引为i的苗框有关类别索引为j的预测

# 在下面，我们定义了这样一个类别预测层，通过参数num_anchors和num_classes分别指定了a(以每个单元为中心生成的描框数)和q(描框类别数) 。 该图层使用填充为1的3x3的卷积层。此卷积层的输入和输出的宽度和高度保持不变。

# 输出的形状是（批量大小，通道数，高度，宽度）
def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3, padding=1)

# 边界框层预测
# 边界框预测层的设计与类别预测层的设计类似。 唯一不同的是，这里需要为每个锚框预测4个偏移量，而不是q+1 个类别
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)

def forward(x, block):
    block.initialize()
    return block(x)

# 连结多尺度的预测
# 正如我们所看到的，除了批量大小这一维度外，其他三个维度都具有不同的尺寸。 为了将这两个预测输出链接起来以提高计算效率，我们将把这些张量转换为更一致的格式。
# 通道维包含中心相同的锚框的预测结果。我们首先将通道维移到最后一维。 因为不同尺度下批量大小仍保持不变，我们可以将预测结果转成二维的（批量大小，高x宽x通道数）的格式，以方便之后在维度1上的连结。
def flatten_pred(pred):
    return npx.batch_flatten(pred.transpose(0, 2, 3, 1))

def concat_preds(preds):
    return np.concatenate([flatten_pred(p) for p in preds], axis=1)

# 高宽减半块
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        # 填充1，3x3卷积层，输出大小不变
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    # 步幅2，2x2最大汇聚层，宽高较少一半
    blk.add(nn.MaxPool2D(2))
    return blk

# 基本网络
# 基本网络块用于从输入图像中抽取特征。 为了计算简洁，我们构造了一个小的基础网络，该网络串联3个高和宽减半块，并逐步将通道数翻倍。 给定输入图像的形状为256x256, 此基本网络块输出的特征图形状为(32x32)
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

print(forward(np.zeros((2, 3, 256, 256)), base_net()).shape)

# 完整网络
# 完整的单发多框检测模型由五个模块组成。每个块生成的特征图既用于生成锚框，又用于预测这些锚框的类别和偏移量。在这五个模块中，第一个是基本网络块，第二个到第四个是高和宽减半块，最后一个模块使用全局最大池将高度和宽度都降到1。从技术上讲，第二到第五个区块都是 图13.7.1中的多尺度特征块。

def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk

# 以图像的每个像素为中心生成不同形状的锚框, 根据缩放数组比和宽高比数组的组合
# 见13.4.1. 生成多个锚框
def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = np.array(sizes, ctx=device)
    ratio_tensor = np.array(ratios, ctx=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长
    steps_w = 1.0 / in_width  # 在x轴上缩放步长

    # 生成锚框的所有中心点
    center_h = (np.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (np.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = np.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

    # 生成“boxes_per_pixel”个高和宽，
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:]))) \
                        * in_height / in_width  # 处理矩形输入
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    # 除以2来获得半高和半宽
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = np.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1).repeat(boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return np.expand_dims(output, axis=0)

# 现在我们为每个块定义前向传播。与图像分类任务不同，此处的输出包括：CNN特征图Y；在当前尺度下根据Y生成的锚框；预测的这些锚框的类别和偏移量（基于Y）。
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    # 得到特征图, 一般是经过一些卷积网络、汇聚层
    # Y的形状（批量大小，通道数，高，宽）
    Y = blk(X)
    # 以每个像素为中心生成N描框, N=len(sizes) + len(ratios) - 1,
    # 总描框total=wxhxN, 详细参考multibox_prior
    # anchors形状(批量大小，描框数，4)，每个描框四个坐标
    # 需要特别注意的是这里的批量大小变成1了，这是有multibox_prior生成函数决定的，
    # 因为每个样本生成的描框的数量和位置完全一样的
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    # print("Y.shape: ", Y.shape)
    # print("anchors.shape: ", anchors.shape)
    # print("anchors: ", anchors[0][:5])
    # 类别预测, 只是经过一层卷积层，输出的形状是（批量大小，通道数，高度，宽度）
    cls_preds = cls_predictor(Y)
    # 边界框预测, 和cls_predictor一样，只是通道数变了
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

def show_anchors(X, anchors, w, h, n):
    print("===show_anchors begin===")
    boxes = anchors[0][:10]
    print("show_anchors 10: ", boxes)
    print("show_anchors shape: ", anchors.shape)
    boxes = anchors.reshape(h, w, n, 4)
    print("show_anchors center shape: ", boxes.shape)
    boxes = boxes[h//2, w//2, :, :]
    print("shwo_anchors center boxes:", boxes)

    # # 用 mxnet.np 创建 CHW: (3, 256, 256)
    # img = np.random.uniform(0, 256, size=(3, h, w)).astype('uint8')
    # # 转成 HWC: (256, 256, 3) 以便 imshow
    # img = img.transpose(1, 2, 0)

    img = X.transpose(1, 2, 0)
    bbox_scale = np.array((w, h, w, h))
    fig = d2l.plt.imshow(img.asnumpy())
    show_bboxes(fig.axes, boxes * bbox_scale)
    print("===show_anchors end===")

# test
# h, w = 561, 728
# X = np.random.uniform(size=(1, 3, h, w))
# Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
# show_anchors(X[0], Y, w, h, 5)

# 数据，因为有五个模块，所以有五组缩放比数组-宽高比数组
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
# 每个中心生成的描框数
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

# 完整模型TinySSD
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.cnt = 0
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            # 每个模块都有自己的类别检测和边界框检测
            # 在每个模块层它们的特征图是不一样的
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            # forward对输入卷积得到前特征图, 然后计算该特征图的描框、预测类别和边界预测
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                    X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                    getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
            # 显示锚框
            if self.cnt == 0:
                print(anchors[i].shape)
                self.cnt += 1
                if i == 1:
                    in_height, in_width = X.shape[-2:]
                    print("X[0].shape:", X[0].shape)
                    # 取前3个通道
                    show_anchors(X[0][:3], anchors=anchors[i], w=in_width, h=in_height, n=4)
        # 合并所有模块的描框(批量大小，描框数，4）-> (批量大小，描框数xn, 4)
        # 需要特别注意的是这里的批量大小变成1了，这是有multibox_prior生成函数决定的，
        anchors = np.concatenate(anchors, axis=1)
        # 合并预测类别(批量大小，通道数，高，宽) -> （批量大小，高x宽x通道数xn）
        # 注意这里xn是指所有模块的相加，
        cls_preds = concat_preds(cls_preds)
        # （批量大小，高x宽x通道数xn） -> （批量大小，-1，类别数+1）
        # 注意类别预测的输出通道数是anchors * (num_classes + 1)，类别预测卷积层指定
        # 因此这里的把（num_classes + 1) 提出来，-1就等于高*宽*anchors*n
        # 注意这里xn是指所有模块的相加，也就是每个模块层的特征图的元素相加
        cls_preds = cls_preds.reshape(
                cls_preds.shape[0], -1, self.num_classes + 1)
        # 合并边界框预测(批量大小，通道数，高，宽) -> （批量大小，高x宽x通道数xn）
        # 注意边界预测的输出通道数是anchors * 4，边界预测卷积层指定
        # 因此实际输出大小是(批量大小，高*宽*anchors*4*n
        # 注意这里xn是指所有模块的相加
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

# 测试
net = TinySSD(num_classes=1)
net.initialize()
X = np.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)
print('测试TinySSD')
print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)

# 训练模型
# 读取数据
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)

# 初始化模型, 只有一个类别，香蕉
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=device)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'wd': 5e-4})

# 定义损失函数和评价函数
# 目标检测有两种类型的损失。 第一种有关锚框类别的损失：我们可以简单地复用之前图像分类问题里一直使用的交叉熵损失函数来计算； 第二种有关正类锚框偏移量的损失：预测偏移量是一个回归问题。 但是，对于这个回归问题，我们在这里不使用 3.1.3节中描述的平方损失，而是使用L1 范数损失，即预测值和真实值之差的绝对值。 掩码变量bbox_masks令负类锚框和填充锚框不参与损失的计算。 最后，我们将锚框类别和偏移量的损失相加，以获得模型的最终损失函数。
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
bbox_loss = gluon.loss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox

# 我们可以沿用准确率评价分类结果。 由于偏移量使用了L1范数损失，我们使用平均绝对误差来评价边界框的预测结果。这些预测结果是从生成的锚框及其预测偏移量中获得的
def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(axis=-1).astype(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

# 训练模型
# 在训练模型时，我们需要在模型的前向传播过程中生成多尺度锚框（anchors），并预测其类别（cls_preds）和偏移量（bbox_preds）。 然后，我们根据标签信息Y为生成的锚框标记类别（cls_labels）和偏移量（bbox_labels）。 最后，我们根据类别和偏移量的预测和标注值计算损失函数。为了代码简洁，这里没有评价测试数据集。
# 这里需要理解的是，本生成的锚框的算法是固定的，不会根据权重的变化而变化，也就是说模型本身不是直接学会画出准备的锚框，而是从预先生成的锚框中根据学习的权重和偏差计算公式修正锚框从而得到正确锚框，其中必须是与真实锚框非常接近的才能学习到更好的参数，所以生成的锚框需要多种缩放比和宽高比
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
for epoch in range(num_epochs):
    # 指标包括：训练精确度的和，训练精确度的和中的示例数，
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    for features, target in train_iter:
        timer.start()
        X = features.as_in_ctx(device)
        Y = target.as_in_ctx(device)
        with autograd.record():
            # 生成多尺度的锚框，为每个锚框预测类别和偏移量
            anchors, cls_preds, bbox_preds = net(X)
            # 为每个锚框标注类别和偏移量
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            # 根据类别和偏移量的预测和标注值计算损失函数
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks);
        l.backward()
        trainer.step(batch_size)
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.size)
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    # 有点问题
    # animator.add(epoch + 1, (cls_err, bbox_mae))

print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter._dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')

# 预测目标
# 在预测阶段，我们希望能把图像里面所有我们感兴趣的目标检测出来。在下面，我们读取并调整测试图像的大小，然后将其转成卷积层需要的四维格式。
img_dir = "/home/yl4946/AI/deep_learn/d2l-zh/mxnet/img/"
img_path = os.path.join(img_dir, "big_banana1.png")
img = image.imread(img_path)
feature = image.imresize(img, 256, 256).astype('float32')
X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)

def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = np.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = np.concatenate((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = scores.argsort()[::-1]
    keep = []  # 保留预测边界框的指标
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = np.nonzero(iou <= iou_threshold)[0]
        B = B[inds + 1]
    return np.array(keep, dtype=np.int32, ctx=boxes.ctx)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.ctx, cls_probs.shape[0]
    anchors = np.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = np.max(cls_prob[1:], 0), np.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = np.arange(num_anchors, dtype=np.int32, ctx=device)
        combined = np.concatenate((keep, all_idx))
        unique, counts = np.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = np.concatenate((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted].astype('float32')
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = np.concatenate((np.expand_dims(class_id, axis=1),
                                np.expand_dims(conf, axis=1),
                                predicted_bb), axis=1)
        out.append(pred_info)
    return np.stack(out)

def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_ctx(device))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)

def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        # if score < threshold:
        #     continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.ctx)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
        # d2l.show_bboxes(fig.axes, bbox)
        print("predict box: ", bbox)

display(img, output, threshold=0.9)

# test
# d2l.set_figsize((5, 5))
# fig = d2l.plt.imshow(img.asnumpy())

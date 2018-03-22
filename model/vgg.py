
# >>>> chainer original VGG meets example's VGG model.
#
# from chainer.links import VGG16Layers
#
# def VGG16():
#     model = VGG16Layers
#     setattr(model, 'insize', 224)
#     return model(pretrained_model=None)
#
#=============================================================================
from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import constant
from chainer.initializers import normal
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.linear import Linear

from functions.accuracy_top_num import accuracy_top_num


# Template of original VGG16

def VGG16class0():
    return VGG16(4096, 1000)

def VGG16class4():
    return VGG16(512, 112)

def VGG16class9():
    return VGG16(512, 12)

class VGG16(chainer.Chain):

    """A VGG-style network for imagenet.

    Original of this model is chainer.links.vision.vgg, and adjust to meet
    our training environments.

    """

    insize = 224

    def __init__(self, num_inter, num_out):
        kwargs = {
            'initialW': normal.Normal(0.01),
            'initial_bias': constant.Zero(),
        }
        self.num_out = num_out
        super(VGG16, self).__init__()

        with self.init_scope():
            self.conv1_1 = Convolution2D(3, 64, 3, 1, 1, **kwargs)
            self.conv1_2 = Convolution2D(64, 64, 3, 1, 1, **kwargs)
            self.conv2_1 = Convolution2D(64, 128, 3, 1, 1, **kwargs)
            self.conv2_2 = Convolution2D(128, 128, 3, 1, 1, **kwargs)
            self.conv3_1 = Convolution2D(128, 256, 3, 1, 1, **kwargs)
            self.conv3_2 = Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv3_3 = Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv4_1 = Convolution2D(256, 512, 3, 1, 1, **kwargs)
            self.conv4_2 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv4_3 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_1 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_2 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_3 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.fc6 = Linear(512 * 7 * 7, num_inter, **kwargs)
            self.fc7 = Linear(num_inter, num_inter, **kwargs)
            self.fc8 = Linear(num_inter, num_out, **kwargs)


    def __call__(self, x, t=None):
        h = x
        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        h = _max_pooling(h)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = _max_pooling(h)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = _max_pooling(h)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = _max_pooling(h)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = _max_pooling(h)

        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        if t is None:
            return F.softmax(h)
        else:
            loss = F.softmax_cross_entropy(h, t)
            top1 = F.accuracy(h,t)
            top5 = accuracy_top_num(h, t, 5)
            chainer.report({'loss': loss, 'accuracy': top1}, self)
            chainer.report({'loss': loss, 'top1': top1, 'top5': top5}, self)
            return loss


# Template of original VGG11

def VGG11class0():
    return VGG11(4096, 1000)

def VGG11class4():
    return VGG11(512, 112)

def VGG11class9():
    return VGG11(512, 12)

class VGG11(chainer.Chain):

    """A VGG-style network for imagenet.

    Original of this model is chainer.links.vision.vgg, and adjust to meet
    our training environments.

    """

    insize = 224

    def __init__(self, num_inter, num_out):
        kwargs = {
            'initialW': normal.Normal(0.01),
            'initial_bias': constant.Zero(),
        }
        super(VGG11, self).__init__()

        with self.init_scope():
            self.conv1_1 = Convolution2D(3, 64, 3, 1, 1, **kwargs)
            self.conv2_1 = Convolution2D(64, 128, 3, 1, 1, **kwargs)
            self.conv3_1 = Convolution2D(128, 256, 3, 1, 1, **kwargs)
            self.conv3_2 = Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv4_1 = Convolution2D(256, 512, 3, 1, 1, **kwargs)
            self.conv4_2 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_1 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_2 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.fc6 = Linear(512 * 7 * 7, num_inter, **kwargs)
            self.fc7 = Linear(num_inter, num_inter, **kwargs)
            self.fc8 = Linear(num_inter, num_out, **kwargs)


    def __call__(self, x, t):
        h = x
        h = F.relu(self.conv1_1(h))
        h = _max_pooling(h)

        h = F.relu(self.conv2_1(h))
        h = _max_pooling(h)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = _max_pooling(h)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = _max_pooling(h)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = _max_pooling(h)

        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h,t)}, self)
        return loss

# Template of VGG16 with BatchNormalization(BN)
# BN layers are inserted only between FCs.

def VGG16BNFCclass0():
    return VGG16BNFC(4096, 1000)

def VGG16BNFCclass4():
    return VGG16BNFC(512, 112)

def VGG16BNFCclass9():
    return VGG16BNFC(512, 12)

class VGG16BNFC(chainer.Chain):

    """A VGG-style network for imagenet.

    Original of this model is chainer.links.vision.vgg, and adjust to meet
    our training environments.

    """

    insize = 224

    def __init__(self, num_inter, num_out):
        kwargs = {
            'initialW': normal.Normal(0.01),
            'initial_bias': constant.Zero(),
        }
        super(VGG16BNFC, self).__init__()

        with self.init_scope():
            self.conv1_1 = Convolution2D(3, 64, 3, 1, 1, **kwargs)
            self.conv1_2 = Convolution2D(64, 64, 3, 1, 1, **kwargs)
            self.conv2_1 = Convolution2D(64, 128, 3, 1, 1, **kwargs)
            self.conv2_2 = Convolution2D(128, 128, 3, 1, 1, **kwargs)
            self.conv3_1 = Convolution2D(128, 256, 3, 1, 1, **kwargs)
            self.conv3_2 = Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv3_3 = Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv4_1 = Convolution2D(256, 512, 3, 1, 1, **kwargs)
            self.conv4_2 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv4_3 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_1 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_2 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_3 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.fc6     = Linear(512 * 7 * 7, num_inter, **kwargs)
            self.bn_fc6  = L.BatchNormalization(num_inter)
            self.fc7     = Linear(num_inter, num_inter, **kwargs)
            self.bn_fc7  = L.BatchNormalization(num_inter)
            self.fc8     = Linear(num_inter, num_out, **kwargs)



    def __call__(self, x, t=None):
        h = x
        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        h = _max_pooling(h)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = _max_pooling(h)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = _max_pooling(h)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = _max_pooling(h)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = _max_pooling(h)

        h = self.fc6(h)
        h = self.bn_fc6(h)
        h = F.relu(h)
        h = F.dropout(h, .5)
        h = self.fc7(h)
        h = self.bn_fc7(h)
        h = F.relu(h)
        h = F.dropout(h, .5)
        h = self.fc8(h)

        if t is None:
            return h
        else:
            loss = F.softmax_cross_entropy(h, t)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h,t)}, self)
            return loss

class VGG16BNFC2(chainer.Chain):

    """A VGG-style network for imagenet.

    Original of this model is chainer.links.vision.vgg, and adjust to meet
    our training environments.

    """

    insize = 224

    def __init__(self, num_inter, num_out):
        kwargs = {
            'initialW': normal.Normal(0.01),
            'initial_bias': constant.Zero(),
        }
        super(VGG16BNFC2, self).__init__()

        with self.init_scope():
            self.conv1_1 = Convolution2D(3, 64, 3, 1, 1, **kwargs)
            self.conv1_2 = Convolution2D(64, 64, 3, 1, 1, **kwargs)
            self.conv2_1 = Convolution2D(64, 128, 3, 1, 1, **kwargs)
            self.conv2_2 = Convolution2D(128, 128, 3, 1, 1, **kwargs)
            self.conv3_1 = Convolution2D(128, 256, 3, 1, 1, **kwargs)
            self.conv3_2 = Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv3_3 = Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv4_1 = Convolution2D(256, 512, 3, 1, 1, **kwargs)
            self.conv4_2 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv4_3 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_1 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_2 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_3 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.bn1     = L.BatchNormalization(512)
            self.fc6     = Linear(512 * 7 * 7, num_inter, **kwargs)
            self.bn2     = L.BatchNormalization(num_inter)
            self.fc7     = Linear(num_inter, num_inter, **kwargs)
            self.bn3     = L.BatchNormalization(num_inter)
            self.fc8     = Linear(num_inter, num_out, **kwargs)



    def __call__(self, x, t=None):
        h = x
        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        h = _max_pooling(h)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = _max_pooling(h)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = _max_pooling(h)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = _max_pooling(h)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = _max_pooling(h)
        h = self.bn1(h)

        h = self.fc6(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, .5)

        h = self.fc7(h)
        h = self.bn3(h)
        h = F.relu(h)
        h = F.dropout(h, .5)

        h = self.fc8(h)

        if t is None:
            return h
        else:
            loss = F.softmax_cross_entropy(h, t)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h,t)}, self)
            return loss

class VGG16BNFC3(chainer.Chain):

    """A VGG-style network for imagenet.

    Original of this model is chainer.links.vision.vgg, and adjust to meet
    our training environments.

    """

    insize = 224

    def __init__(self, num_inter, num_out, dropout_ratio=.5):
        kwargs = {
            'initialW': normal.Normal(0.01),
            'initial_bias': constant.Zero(),
        }
        super(VGG16BNFC3, self).__init__()

        with self.init_scope():
            self.conv1_1 = Convolution2D(3, 64, 3, 1, 1, **kwargs)
            self.conv1_2 = Convolution2D(64, 64, 3, 1, 1, **kwargs)
            self.conv2_1 = Convolution2D(64, 128, 3, 1, 1, **kwargs)
            self.conv2_2 = Convolution2D(128, 128, 3, 1, 1, **kwargs)
            self.conv3_1 = Convolution2D(128, 256, 3, 1, 1, **kwargs)
            self.conv3_2 = Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv3_3 = Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv4_1 = Convolution2D(256, 512, 3, 1, 1, **kwargs)
            self.conv4_2 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv4_3 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_1 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_2 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_3 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.bn1     = L.BatchNormalization(512)
            self.fc6     = Linear(512 * 7 * 7, num_inter, **kwargs)
            self.bn2     = L.BatchNormalization(num_inter)
            self.fc7     = Linear(num_inter, num_inter, **kwargs)
            self.bn3     = L.BatchNormalization(num_inter)
            self.fc8     = Linear(num_inter, num_out, **kwargs)
        self.dropout_ratio = dropout_ratio



    def __call__(self, x, t=None):
        h = x
        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        h = _max_pooling(h)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = _max_pooling(h)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = _max_pooling(h)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = _max_pooling(h)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))

        # conv. layers are freezed(Use as is, don't change)
        # dropout and batch-norm layers are for full connectedly layers.
        h = self.conv5_3(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = _max_pooling(h)

        h = self.fc6(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout_ratio)

        h = self.fc7(h)
        h = self.bn3(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout_ratio)

        h = self.fc8(h)

        if t is None:
            return h
        else:
            loss = F.softmax_cross_entropy(h, t)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h,t)}, self)
            return loss


# Template of VGG16 with BatchNormalization(BN)
# BN layers are inserted among both conv. and fc.

def VGG16BNclass0():
    return VGG16BN(4096, 1000)

def VGG16BNclass4():
    return VGG16BN(512, 112)

def VGG16BNclass9():
    return VGG16BN(512, 12)


""" Following class 'Block' and 'VGGBN' is copied from
the examples/cifar/model/VGG.py of chainer/chainer repository.
'Block' is not modified(i.e. the class is equal to original one).
'VGGBN' is modified. Original name is 'VGG'. """

class Block(chainer.Chain):

    """A convolution, batch norm, ReLU block.

    A block in a feedforward network that performs a
    convolution followed by batch normalization followed
    by a ReLU activation.

    For the convolution operation, a square filter size is used.

    Args:
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self, out_channels, ksize, pad=1):
        kwargs = {
            'initialW': normal.Normal(0.01),
            'initial_bias': constant.Zero(),
        }
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad, **kwargs)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)

class VGG16BN(chainer.Chain):

    """A VGG-style network for very small images.

    This model is based on the VGG-style model from
    http://torch.ch/blog/2015/07/30/cifar.html
    which is based on the network architecture from the paper:
    https://arxiv.org/pdf/1409.1556v6.pdf

    This model is intended to be used with either RGB or greyscale input
    images that are of size 32x32 pixels, such as those in the CIFAR10
    and CIFAR100 datasets.

    On CIFAR10, it achieves approximately 89% accuracy on the test set with
    no data augmentation.

    On CIFAR100, it achieves approximately 63% accuracy on the test set with
    no data augmentation.

    Args:
        class_labels (int): The number of class labels.

    """

    insize = 224

    def __init__(self, num_inter, num_out):
        kwargs = {
            'initialW': normal.Normal(0.01),
            'initial_bias': constant.Zero(),
        }
        super(VGG16BN, self).__init__()
        with self.init_scope():
            self.block1_1 = Block(64, 3)
            self.block1_2 = Block(64, 3)
            self.block2_1 = Block(128, 3)
            self.block2_2 = Block(128, 3)
            self.block3_1 = Block(256, 3)
            self.block3_2 = Block(256, 3)
            self.block3_3 = Block(256, 3)
            self.block4_1 = Block(512, 3)
            self.block4_2 = Block(512, 3)
            self.block4_3 = Block(512, 3)
            self.block5_1 = Block(512, 3)
            self.block5_2 = Block(512, 3)
            self.block5_3 = Block(512, 3)
            self.fc1 = L.Linear(None, num_inter, **kwargs)
            self.bn_fc1 = L.BatchNormalization(num_inter)
            self.fc2 = L.Linear(None, num_inter, **kwargs)
            self.bn_fc2 = L.BatchNormalization(num_inter)
            self.fc3 = L.Linear(None, num_out, **kwargs)


    def __call__(self, x, t):
        # 64 channel blocks:
        h = x
        h = self.block1_1(h)
        h = F.dropout(h, .3)
        h = self.block1_2(h)
        h = _max_pooling(h)

        # 128 channel blocks:
        h = self.block2_1(h)
        h = F.dropout(h, .4)
        h = self.block2_2(h)
        h = _max_pooling(h)

        # 256 channel blocks:
        h = self.block3_1(h)
        h = F.dropout(h, .4)
        h = self.block3_2(h)
        h = F.dropout(h, .4)
        h = self.block3_3(h)
        h = _max_pooling(h)

        # 512 channel blocks:
        h = self.block4_1(h)
        h = F.dropout(h, .4)
        h = self.block4_2(h)
        h = F.dropout(h, .4)
        h = self.block4_3(h)
        h = _max_pooling(h)

        # 512 channel blocks:
        h = self.block5_1(h)
        h = F.dropout(h, .4)
        h = self.block5_2(h)
        h = F.dropout(h, .4)
        h = self.block5_3(h)
        h = _max_pooling(h)

        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = F.relu(h)
        h = F.dropout(h, .5)
        h = self.fc2(h)
        h = self.bn_fc2(h)
        h = F.relu(h)
        h = F.dropout(h, .5)
        h = self.fc3(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h,t)}, self)
        return loss

def _max_pooling(h):
    return F.max_pooling_2d(h, 2)

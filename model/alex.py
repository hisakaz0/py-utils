import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

"""
NOTE: Syntax of this program are updated from v1 to v2 of chainer!!

This program is coded within v1 when this program is provided.
"""



class Alex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self):

        # maybe original initializer
        initialW = initializers.Normal()

        super(Alex, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3,   96, 11, stride=4, initialW=initialW)
            self.conv2 = L.Convolution2D(96,  256,  5, pad=2, initialW=initialW)
            self.conv3 = L.Convolution2D(256, 384,  3, pad=1, initialW=initialW)
            self.conv4 = L.Convolution2D(384, 384,  3, pad=1, initialW=initialW)
            self.conv5 = L.Convolution2D(384, 256,  3, pad=1, initialW=initialW)
            self.fc6 = L.Linear(256,  4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, 1000)

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss


class AlexVal(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227
    layer_names = [
            'conv1', 'relu1', 'bn1',  'pool1',
            'conv2', 'relu2', 'bn2',  'pool2',
            'conv3', 'relu3',
            'conv4', 'relu4',
            'conv5', 'relu5', 'pool5',
            'fc6',  'relu6', 'dp6',
            'fc7',  'relu7', 'dp7',
            'fc8', 'prob']

    def __init__(self, layers=None):
        super(AlexVal, self).__init__(
            conv1=L.Convolution2D(None,  96, 11, stride=4),
            conv2=L.Convolution2D(None, 256,  5, pad=2),
            conv3=L.Convolution2D(None, 384,  3, pad=1),
            conv4=L.Convolution2D(None, 384,  3, pad=1),
            conv5=L.Convolution2D(None, 256,  3, pad=1),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 4096),
            fc8=L.Linear(None, 1000),
        )
        self.train = False
        self._check_layer_name(layers)

    def __call__(self, x):

        h = {}
        h['conv1'] = self.conv1(x)
        h['relu1'] = F.relu(h['conv1'])
        h['bn1']   = F.local_response_normalization(h['relu1'])
        h['pool1'] = F.max_pooling_2d(h['bn1'], 3, stride=2)
        h['conv2'] = self.conv2(h['pool1'])
        h['relu2'] = F.relu(h['conv2'])
        h['bn2']   = F.local_response_normalization(h['relu2'])
        h['pool2'] = F.max_pooling_2d(h['bn2'], 3, stride=2)
        h['conv3'] = self.conv3(h['pool2'])
        h['relu3'] = F.relu(h['conv3'])
        h['conv4'] = self.conv4(h['relu3'])
        h['relu4'] = F.relu(h['conv4'])
        h['conv5'] = self.conv5(h['relu4'])
        h['relu5'] = F.relu(h['conv5'])
        h['pool5'] = F.max_pooling_2d(h['relu5'], 3, stride=2)
        h['fc6']   = self.fc6(h['pool5'])
        h['relu6'] = F.relu(h['fc6'])
        h['dp6']   = F.dropout(h['relu6'])
        h['fc7']   = self.fc7(h['dp6'])
        h['relu7'] = F.relu(h['fc7'])
        h['dp7']   = F.dropout(h['relu7'])
        h['fc8']   = self.fc8(h['dp7'])
        h['prob']  = F.softmax(h['fc8'])
        return {k:v for (k,v) in h.items() if k in self.layers}

    def _check_layer_name(self, layers):
        if layers is None:
            layers = ['prob']
        elif type(layers) != type([]):
            raise TypeError("type of 'outputs' must be list class")

        for l in layers:
            assert l in self.layer_names, "Invalid layer name: {}".format(l)
        self.layers = layers


class AlexValPartial(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227
    layer_names = [
            'conv1', 'relu1', 'bn1',  'pool1',
            'conv2', 'relu2', 'bn2',  'pool2',
            'conv3', 'relu3',
            'conv4', 'relu4',
            'conv5', 'relu5', 'pool5',
            'fc6',  'relu6', 'dp6',
            'fc7',  'relu7', 'dp7',
            'fc8',
            'prob']

    def __init__(self, layer='fc6'):
        super(AlexValPartial, self).__init__(
            conv1=L.Convolution2D(None,  96, 11, stride=4),
            conv2=L.Convolution2D(None, 256,  5, pad=2),
            conv3=L.Convolution2D(None, 384,  3, pad=1),
            conv4=L.Convolution2D(None, 384,  3, pad=1),
            conv5=L.Convolution2D(None, 256,  3, pad=1),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 4096),
            fc8=L.Linear(None, 1000),
        )
        self.train = False
        self.layer = self._check_layer_name(layer)

    def __call__(self, x):

        # TODO: to flexible

        h = {}
        # h['conv1'] = self.conv1(x)
        # h['relu1'] = F.relu(h['conv1'])
        # h['bn1']   = F.local_response_normalization(h['relu1'])
        # h['pool1'] = F.max_pooling_2d(h['bn1'], 3, stride=2)

        if self.layer == 'pool1':
            h['conv2'] = self.conv2(x)
            h['relu2'] = F.relu(h['conv2'])
            h['bn2']   = F.local_response_normalization(h['relu2'])
            h['pool2'] = F.max_pooling_2d(h['bn2'], 3, stride=2)
            h['conv3'] = self.conv3(h['pool2'])
            h['relu3'] = F.relu(h['conv3'])
            h['conv4'] = self.conv4(h['relu3'])
            h['relu4'] = F.relu(h['conv4'])
            h['conv5'] = self.conv5(h['relu4'])
            h['relu5'] = F.relu(h['conv5'])
            h['pool5'] = F.max_pooling_2d(h['relu5'], 3, stride=2)
            h['fc6']   = self.fc6(h['pool5'])
            h['relu6'] = F.relu(h['fc6'])
            h['dp6']   = F.dropout(h['relu6'])
            h['fc7']   = self.fc7(h['dp6'])
            h['relu7'] = F.relu(h['fc7'])
            h['dp7']   = F.dropout(h['relu7'])
            h['fc8']   = self.fc8(h['dp7'])
        elif self.layer == 'pool2':
            h['conv3'] = self.conv3(x)
            h['relu3'] = F.relu(h['conv3'])
            h['conv4'] = self.conv4(h['relu3'])
            h['relu4'] = F.relu(h['conv4'])
            h['conv5'] = self.conv5(h['relu4'])
            h['relu5'] = F.relu(h['conv5'])
            h['pool5'] = F.max_pooling_2d(h['relu5'], 3, stride=2)
            h['fc6']   = self.fc6(h['pool5'])
            h['relu6'] = F.relu(h['fc6'])
            h['dp6']   = F.dropout(h['relu6'])
            h['fc7']   = self.fc7(h['dp6'])
            h['relu7'] = F.relu(h['fc7'])
            h['dp7']   = F.dropout(h['relu7'])
            h['fc8']   = self.fc8(h['dp7'])
        elif self.layer == 'pool5':
            h['fc6']   = self.fc6(x)
            h['relu6'] = F.relu(h['fc6'])
            h['dp6']   = F.dropout(h['relu6'])
            h['fc7']   = self.fc7(h['dp6'])
            h['relu7'] = F.relu(h['fc7'])
            h['dp7']   = F.dropout(h['relu7'])
            h['fc8']   = self.fc8(h['dp7'])
        else:
            raise NameError("Invalid layer name: {}" .format(self.layer))

        return h['fc8']

    def _check_layer_name(self, layer):
        if type(layer) != type(''):
            raise TypeError("type of 'outputs' must be str class")
        assert layer in self.layer_names, "Invalid layer name"
        return layer


class AlexFp16(Alex):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self):
        chainer.Chain.__init__(self)
        self.dtype = np.float16
        W = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        bias = initializers.Zero(self.dtype)

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, 11, stride=4,
                                         initialW=W, initial_bias=bias)
            self.conv2 = L.Convolution2D(None, 256, 5, pad=2,
                                         initialW=W, initial_bias=bias)
            self.conv3 = L.Convolution2D(None, 384, 3, pad=1,
                                         initialW=W, initial_bias=bias)
            self.conv4 = L.Convolution2D(None, 384, 3, pad=1,
                                         initialW=W, initial_bias=bias)
            self.conv5 = L.Convolution2D(None, 256, 3, pad=1,
                                         initialW=W, initial_bias=bias)
            self.fc6 = L.Linear(None, 4096, initialW=W, initial_bias=bias)
            self.fc7 = L.Linear(None, 4096, initialW=W, initial_bias=bias)
            self.fc8 = L.Linear(None, 1000, initialW=W, initial_bias=bias)

    def __call__(self, x, t):
        return Alex.__call__(self, F.cast(x, self.dtype), t)


# def copy_caffe_model(caffe_model, layer_list):
#     model = Alex()
#     assert type(layer_list) == type([]), \
#             "Invalid type of layer_list. must be 'list'"
#     for layer in layer_list:
#         assert layer in caffe_model._children, \
#                 "caffe model does not have the layer, '{}'". format(layer)
#         assert layer in model._children, \
#                 "chainer model does not have the layer, '{}'". format(layer)
#         model[layer].initialW = caffe_model[layer].W.data
#         model[layer].b.data   = caffe_model[layer].b.data





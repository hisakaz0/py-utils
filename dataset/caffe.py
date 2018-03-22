

import numpy
from numpy.random import randint
from chainer.dataset import DatasetMixin
from chainer.datasets import LabeledImageDataset
from dataset.multi_labeled_image_dataset import MultiLabeledImageDataset

from dataset.Dataset import resize_numpy_image
from dataset.Dataset import mirrored_image
from dataset.Dataset import IndexExtractor
from dataset.Dataset import calculate_rectangle_image
from dataset.Dataset import convert_grayscale_rgb

class CaffePreprocessedDataset(DatasetMixin):
    def __init__(self, pair, root, transformer):
        self.base = LabeledImageDataset(pair, root)
        self.transformer = transformer

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]

        # convert from gray-scale to colored
        c, h, w = image.shape
        if c is 1:
            _image = numpy.ndarray((3, h, w))
            _image[:] = image
            image = _image

        # convert a image representation from chainer-like to caffe-like
        image = image.transpose(1, 2, 0) # convert original
        image *= (1.0/255.0) # scaled to [0,1]

        # transform the image
        layer_name = list(self.transformer.inputs.keys())[0]
        image = self.transformer.preprocess(layer_name, image)

        return image, label

class CaffePreprocessedDataset2(DatasetMixin):
    """self.base is LabeledImageDataset. This dataset supports data-argumentation using 
    random cropping and random flipping horizontaly. Preprocess procedures are following items.
    1. Random cropping, random flipping
    2. Caffe transforming
    """ 

    def __init__(self, pair, root, transformer, crop_size, random=True):
        self.base = LabeledImageDataset(pair, root)
        self.transformer = transformer
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        crop_size = self.crop_size

        # convert from gray-scale to colored
        c, h, w = image.shape
        if c is 1:
            _image = numpy.ndarray((3, h, w))
            _image[:] = image
            image = _image

        # cropping
        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size
        image = image[:, top:bottom, left:right]

        # convert a image representation from chainer-like to caffe-like
        image = image.transpose(1, 2, 0) # convert original
        image *= (1.0/255.0) # scaled to [0,1]

        # transform the image
        layer_name = list(self.transformer.inputs.keys())[0]
        image = self.transformer.preprocess(layer_name, image)

        return image, label

class CaffePreprocessedDataset3(DatasetMixin):
    """MultiLabeledDataset"""

    def __init__(self, pair, root, transformer, crop_size, random=True):
        self.base = MultiLabeledImageDataset(pair, root)
        self.transformer = transformer
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        crop_size = self.crop_size

        # convert from gray-scale to colored
        c, h, w = image.shape
        if c is 1:
            _image = numpy.ndarray((3, h, w))
            _image[:] = image
            image = _image

        # cropping
        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size
        image = image[:, top:bottom, left:right]

        # convert a image representation from chainer-like to caffe-like
        image = image.transpose(1, 2, 0) # convert original
        image *= (1.0/255.0) # scaled to [0,1]

        # transform the image
        layer_name = list(self.transformer.inputs.keys())[0]
        image = self.transformer.preprocess(layer_name, image)

        return image, label

class CaffePreprocessedDataset4(DatasetMixin):
    """Single crop"""

    def __init__(self, pair, root, crop_size, scales, mean, random=True):
        # ここでのrandomは簡易validationをやるため
        # ガチのtestはそれぞれのclassを用意する

        self.base = LabeledImageDataset(pair, root)
        self.crop_size = crop_size
        self.scales = scales
        self.mean = mean
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        crop_size = self.crop_size
        scales = self.scales

        # convert from gray-scale to dammy colored
        image = convert_grayscale_rgb(image)

        # resize
        def resize_rectangle(image, scales):
            scale_index = randint(0, len(scales))
            new_h, new_w = calculate_rectangle_image(image, scales[scale_index])
            image = resize_numpy_image(image, new_h, new_w)
            return image
        image = resize_rectangle(image, scales)
        _, h, w = image.shape

        # cropping
        if self.random:
            # Randomly crop a region and flip the image
            top = randint(0, h - crop_size)
            left = randint(0, w - crop_size)
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size
        image = image[:, top:bottom, left:right]

        # channel swap
        channel_swap = (2, 1, 0)
        image = image[channel_swap, :, :] # RGB -> BGR

        # subtract mean
        image = image.transpose((1, 2, 0))
        image -= self.mean.astype(numpy.float32)
        image = image.transpose((2, 0, 1))

        # mirror
        if randint(0, 2):
            image = mirrored_image(image)

        return image, label

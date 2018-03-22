
"""Main functions of dataset module"""

from chainer.datasets.image_dataset import _read_image_as_array
from chainer.dataset import DatasetMixin
from chainer.datasets import LabeledImageDataset
import numpy
from PIL import Image
from numpy import random


def numpy_image(path, dtype=numpy.float32):
    """ Create numpy image
    when `path` to a image is opened, then read with `PIL.Image` moduel.
    After than, the image is converted to `numpy.ndarray` format."""
    image = _read_image_as_array(path, dtype)
    if image.ndim == 2:
        # image is greyscale
        image = image[:, :, numpy.newaxis]
    return image.transpose(2, 0, 1)

def preprocessed_image(image, mean, top, left, bottom, right):
    """Pre-processed image
    return the preprocessed image(cropped, and subtracted mean)."""
    assert mean.shape == image.shape, \
        "A shape of mean is difference with a shape of image"
    return image[:, top:bottom, left:right] - mean[:, top:bottom, left:right]


def mirrored_image(image):
    """Mirror Image
    This function return a image which is mirrored horizontally."""
    return image[:, :, ::-1]

def calculate_rectangle_image(image, size):
    """Calculate rectangle image.
    Calculate new height and width which are resize with `size`.
    A shorter size of image is `size`. On the other hand, a
    longer size is resized with keeping the original aspect ratio."""
    _, h, w = image.shape

    if h > w:
        shorter_side = w
    else:
        shorter_side = h
    resize_ratio = float(size) / shorter_side

    return round(h * resize_ratio), round(w * resize_ratio)

def convert_grayscale_rgb(image):
    """Grayscale to RGB-colored image
    `image` must be `numpy.ndarray`."""
    c, h, w = image.shape
    if c is 1:
        _image = numpy.ndarray((3, h, w))
        _image[:] = image
        image = _image
    return image

def center_crop(image):
    """Center crop input image
    `image` is `numpy.ndarray`."""
    _, h, w = image.shape

    if h > w:
        top, left = (h - w) // 2, 0
        bottom, right = top + w, left + w
    else:
        top, left = 0, (w - h) // 2
        bottom, right = top + h, left + h
    return image[:, top:bottom, left:right]

def channel_swap(image, channel):
    """Channel swap
    It is used to convert RGB to BGR."""
    return image[channel, :, :]

class IndexExtractor:

    def __init__(self, d, first_key='image'):
        """Example
        >>> d = {
        >>>         'horizontal': 5,
        >>>         'vertical': 5,
        >>>         'mirror': 2,
        >>>         'scales': 3 }

        >>> extractor = IndexExtractor(d)

        >>> for i in range(147, 152):
        >>>     print("{}: {}". format(i, extractor(i)))
        147: {'image': 0, 'scales': 2, 'mirror': 1, 'vertical': 4, 'horizontal': 2}
        148: {'image': 0, 'scales': 2, 'mirror': 1, 'vertical': 4, 'horizontal': 3}
        149: {'image': 0, 'scales': 2, 'mirror': 1, 'vertical': 4, 'horizontal': 4}
        150: {'image': 1, 'scales': 0, 'mirror': 0, 'vertical': 0, 'horizontal': 0}
        151: {'image': 1, 'scales': 0, 'mirror': 0, 'vertical': 0, 'horizontal': 1}

        >>> print(len(extractor))
        150"""

        self.d = {}
        for key, value in d.items():
            assert type(key) is str, \
                    "key type is not str, but {}". format(type(key))
            assert type(value) is int, \
                    "value type is not int, but {}". format(type(value))
            self.d[key] = value
        self.first_key = first_key

    def __call__(self, index, d=None, r=None):
        if d is None:
            d = self.d.copy()
            key = self.first_key
        else:
            key, _ = d.popitem()
        divisor = 1
        for v in d.values():
            divisor *= v
        key_index = index // divisor
        index = index - key_index * divisor
        if r is None:
            r = {}
        r[key] = key_index
        if len(d) is 0:
            return r
        else:
            return self.__call__(index, d, r)

    def __len__(self):
        l = 1
        for v in self.d.values():
            l *= v
        return l

def resize_numpy_image(image, h, w):
    c, _, _ = image.shape
    image = image.transpose((1, 2, 0)) # C,H,W -> H,W,C
    image = Image.fromarray(image.astype(numpy.uint8))
    image = image.convert('RGB')
    image = image.resize((w, h))
    image = numpy.asarray(image, dtype=numpy.float32) # PIL -> numpy
    return image.transpose((2, 0, 1)) # H,W,C -> C,H,W


class PreprocessedDataset(DatasetMixin):
    def __init__(self, path, root, mean, crop_size, random=True, normalize=False):
        self.base = LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random
        self.normalize = normalize

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        c, h, w = image.shape

        if c is 1: # gray scale
            _image = numpy.ndarray((3, h, w))
            _image[:] = image
            image = _image

        if image.dtype is not self.mean.dtype: # convert dtype
            image = image.astype('f')

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
        image -= self.mean[:, top:bottom, left:right]

        if self.normalize:
            image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


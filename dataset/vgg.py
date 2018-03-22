
import numpy
from chainer.dataset import DatasetMixin
from chainer.datasets import LabeledImageDataset

from dataset.Dataset import mirrored_image
from dataset.Dataset import preprocessed_image
from dataset.multi_labeled_image_dataset import MultiLabeledImageDataset


class VGGTestDataset(DatasetMixin):
    """Dataset for testing of VGG
    `get_example()` of this dataset return 50 croped input images. The way to
    crop is described in the original paper."""

    NUM_CROP = 50,
    NUM_MIRROR = 2,
    NUM_HORIZONTAL_GRID = 5,
    NUM_VERTICAL_GRID = 5,

    def __init__(self, path, root, mean, crop_size):
        self.base = LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size


    def __len__(self):
        return len(self.base) * self.NUM_CROP[0]


    def _extract_index(self, index):
        _num_crop = self.NUM_CROP[0]
        _num_mirror = self.NUM_MIRROR[0]
        _num_vertical = self.NUM_VERTICAL_GRID[0]
        _num_horizontal = self.NUM_HORIZONTAL_GRID[0]

        base = index // _num_crop
        index = index - base * _num_crop
        is_mirror = index // (_num_horizontal * _num_vertical)
        index = index - is_mirror * (_num_horizontal* _num_vertical)
        vertical = index // _num_horizontal
        horizontal = index - vertical * _num_horizontal
        return (base, is_mirror, vertical, horizontal)


    def _get_grid_step(self, image_size, num_grid):
        return (image_size - self.crop_size) // (num_grid - 1)


    def _get_crop_box(self, num_horizontal, h_grid_step, num_vertical,
            w_grid_step):
        top = num_horizontal * h_grid_step
        bottom = top + self.crop_size
        left = num_vertical * w_grid_step
        right = left + self.crop_size
        return (top, left, bottom, right)


    def get_example(self, i):

        base, vertical, horizontal, is_mirror = self._extract_index(i)

        crop_size = self.crop_size
        image, label = self.base[base]

        _, h, w = image.shape

        h_grid_step = self._get_grid_step(h, self.NUM_HORIZONTAL_GRID[0])
        w_grid_step = self._get_grid_step(w, self.NUM_VERTICAL_GRID[0])

        image = preprocessed_image(image, self.mean,
                *self._get_crop_box(horizontal, h_grid_step, vertical,
                    w_grid_step))

        if is_mirror:
            image = mirrored_image(image)

        return (image, label)

class VGGCaffeTestDataset(DatasetMixin):
    """Dataset for testing of VGG
    `get_example()` of this dataset return 50 croped input images. The way to
    crop is described in the original paper."""

    NUM_CROP = 50,
    NUM_MIRROR = 2,
    NUM_HORIZONTAL_GRID = 5,
    NUM_VERTICAL_GRID = 5,

    def __init__(self, path, root, transformer, crop_size):
        self.base = LabeledImageDataset(path, root)
        self.transformer = transformer
        self.crop_size = crop_size


    def __len__(self):
        return len(self.base) * self.NUM_CROP[0]


    def _extract_index(self, index):
        _num_crop = self.NUM_CROP[0]
        _num_mirror = self.NUM_MIRROR[0]
        _num_vertical = self.NUM_VERTICAL_GRID[0]
        _num_horizontal = self.NUM_HORIZONTAL_GRID[0]

        base = index // _num_crop
        index = index - base * _num_crop
        is_mirror = index // (_num_horizontal * _num_vertical)
        index = index - is_mirror * (_num_horizontal* _num_vertical)
        vertical = index // _num_horizontal
        horizontal = index - vertical * _num_horizontal
        return (base, is_mirror, vertical, horizontal)


    def _get_grid_step(self, image_size, num_grid):
        return (image_size - self.crop_size) // (num_grid - 1)


    def _get_crop_box(self, num_horizontal, h_grid_step, num_vertical,
            w_grid_step):
        top = num_horizontal * h_grid_step
        bottom = top + self.crop_size
        left = num_vertical * w_grid_step
        right = left + self.crop_size
        return (top, left, bottom, right)


    def get_example(self, i):

        base, vertical, horizontal, is_mirror = self._extract_index(i)

        crop_size = self.crop_size
        image, label = self.base[base]

        _, h, w = image.shape

        # get crop position
        h_grid_step = self._get_grid_step(h, self.NUM_HORIZONTAL_GRID[0])
        w_grid_step = self._get_grid_step(w, self.NUM_VERTICAL_GRID[0])

        # crop 224x224
        top, left, bottom, right = \
                self._get_crop_box(horizontal, h_grid_step, vertical, w_grid_step)
        image = image[:, top:bottom, left:right]

        # convert from gray-scale to colored
        c, h, w = image.shape
        if c is 1:
            _image = numpy.ndarray((3, h, w))
            _image[:] = image
            image = _image

        # convert a image representation from chainer-like to caffe-like
        image = image.transpose(1, 2, 0) # convert original
        image *= (1.0/255.0) # scaled to [0,1]

        # transform the image(caffe)
        # NOTE: transformerでresize関数が呼ばれるが、既に224x224でcropしている
        # ので、拡大･縮小はされない
        layer_name = list(self.transformer.inputs.keys())[0]
        image = self.transformer.preprocess(layer_name, image)

        if is_mirror:
            image = mirrored_image(image)

        return (image, label)

class VGGMultiLabeledCaffeTestDataset(DatasetMixin):
    """Dataset for testing of VGG
    `get_example()` of this dataset return 50 croped input images. The way to
    crop is described in the original paper."""

    NUM_CROP = 50,
    NUM_MIRROR = 2,
    NUM_HORIZONTAL_GRID = 5,
    NUM_VERTICAL_GRID = 5,

    def __init__(self, path, root, transformer, crop_size):
        self.base = MultiLabeledImageDataset(path, root)
        self.transformer = transformer
        self.crop_size = crop_size


    def __len__(self):
        return len(self.base) * self.NUM_CROP[0]


    def _extract_index(self, index):
        _num_crop = self.NUM_CROP[0]
        _num_mirror = self.NUM_MIRROR[0]
        _num_vertical = self.NUM_VERTICAL_GRID[0]
        _num_horizontal = self.NUM_HORIZONTAL_GRID[0]

        base = index // _num_crop
        index = index - base * _num_crop
        is_mirror = index // (_num_horizontal * _num_vertical)
        index = index - is_mirror * (_num_horizontal* _num_vertical)
        vertical = index // _num_horizontal
        horizontal = index - vertical * _num_horizontal
        return (base, is_mirror, vertical, horizontal)


    def _get_grid_step(self, image_size, num_grid):
        return (image_size - self.crop_size) // (num_grid - 1)


    def _get_crop_box(self, num_horizontal, h_grid_step, num_vertical,
            w_grid_step):
        top = num_horizontal * h_grid_step
        bottom = top + self.crop_size
        left = num_vertical * w_grid_step
        right = left + self.crop_size
        return (top, left, bottom, right)


    def get_example(self, i):

        base, vertical, horizontal, is_mirror = self._extract_index(i)

        crop_size = self.crop_size
        image, label = self.base[base]

        _, h, w = image.shape

        # get crop position
        h_grid_step = self._get_grid_step(h, self.NUM_HORIZONTAL_GRID[0])
        w_grid_step = self._get_grid_step(w, self.NUM_VERTICAL_GRID[0])

        # crop 224x224
        top, left, bottom, right = \
                self._get_crop_box(horizontal, h_grid_step, vertical, w_grid_step)
        image = image[:, top:bottom, left:right]

        # convert from gray-scale to colored
        c, h, w = image.shape
        if c is 1:
            _image = numpy.ndarray((3, h, w))
            _image[:] = image
            image = _image

        # convert a image representation from chainer-like to caffe-like
        image = image.transpose(1, 2, 0) # convert original
        image *= (1.0/255.0) # scaled to [0,1]

        # transform the image(caffe)
        # NOTE: transformerでresize関数が呼ばれるが、既に224x224でcropしている
        # ので、拡大･縮小はされない
        layer_name = list(self.transformer.inputs.keys())[0]
        image = self.transformer.preprocess(layer_name, image)

        if is_mirror:
            image = mirrored_image(image)

        return (image, label)


import numpy
from chainer.dataset import DatasetMixin
from chainer.datasets import LabeledImageDataset

from dataset.multi_labeled_image_dataset import MultiLabeledImageDataset
from dataset.Dataset import resize_numpy_image
from dataset.Dataset import mirrored_image
from dataset.Dataset import IndexExtractor
from dataset.Dataset import calculate_rectangle_image
from dataset.Dataset import convert_grayscale_rgb

class VGGTestDataset(DatasetMixin):
    """Dataset for testing of VGG
    `get_example()` of this dataset return 50 croped input images. The way to
    crop is described in the original paper."""

    def __init__(self, path, root, mean, crop_size=224, scales=[256, 384, 512]):
        self.base = LabeledImageDataset(path, root)
        self.mean = mean # BGR
        self.crop_size = crop_size
        self.scales = scales
        self.process_dict = {
                'horizontal': 5,
                'vertical': 5,
                'mirror': 2,
                'scales': len(scales)}
        self.index_extractor = IndexExtractor(self.process_dict)

    def __len__(self):
        return len(self.base) * len(self.index_extractor)

    @property
    def num_crop(self):
        return len(self.index_extractor)

    def _get_grid_step(self, image_size, num_grid):
        return (image_size - self.crop_size) // (num_grid - 1)

    def _get_crop_box(self, num_horizontal, h_grid_step,
            num_vertical, w_grid_step):
        top = num_horizontal * h_grid_step
        bottom = top + self.crop_size
        left = num_vertical * w_grid_step
        right = left + self.crop_size
        return (top, left, bottom, right)

    def get_example(self, i):
        process_index = self.index_extractor(i)
        crop_size = self.crop_size
        image, label = self.base[process_index['image']]

        # convert from gray-scale to dammy colored
        image = convert_grayscale_rgb(image)

        # resize
        new_h, new_w = calculate_rectangle_image(
                image, self.scales[process_index['scales']])
        image = resize_numpy_image(image, new_h, new_w)
        _, h, w = image.shape

        # cropping
        h_grid_step = self._get_grid_step(h, self.process_dict['horizontal'])
        w_grid_step = self._get_grid_step(w, self.process_dict['vertical'])
        top, left, bottom, right = self._get_crop_box(
                process_index['horizontal'], h_grid_step,
                process_index['vertical'], w_grid_step)
        image = image[:, top:bottom, left:right]

        # channel swap
        channel_swap = (2, 1, 0)
        image = image[channel_swap, :, :] # RGB -> BGR

        # subtract mean
        image = image.transpose((1, 2, 0))
        image -= self.mean.astype(numpy.float32)
        image = image.transpose((2, 0, 1))

        # mirror
        if process_index['mirror']:
            image = mirrored_image(image)

        return image, label


class VGGMultiLabeledTestDataset(VGGTestDataset):

    def __init__(self, path, root, mean, crop_size=224, scales=[256, 384, 512]):
        self.base = MultiLabeledImageDataset(path, root)
        self.mean = mean # BGR
        self.crop_size = crop_size
        self.scales = scales
        self.process_dict = {
                'horizontal': 5,
                'vertical': 5,
                'mirror': 2,
                'scales': len(scales)}
        self.index_extractor = IndexExtractor(self.process_dict)

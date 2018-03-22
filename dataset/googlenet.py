
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


class GooglenetTestDataset(DatasetMixin):

    def __init__(self, path, root, mean, crop_size=224,
            scales=[256, 288, 320, 352]):
        self.base = LabeledImageDataset(path, root)
        self.mean = mean # BGR
        self.crop_size = crop_size
        self.scales = scales
        self.process_dict = {
                '1st_preprocess': 3, # left, center, right cropping
                '2nd_preprocess': 6, # 4-corner and center cropping, resize
                'mirror': 2,
                'scales': len(scales)}
        self.index_extractor = IndexExtractor(self.process_dict)

    @property
    def num_crop(self):
        return len(self.index_extractor)

    def __len__(self):
        return len(self.base) * self.num_crop

    def image_preprocess(self, image, process_index):
        crop_size = self.crop_size

        # grayscale to colored image
        image = convert_grayscale_rgb(image)

        # resize
        new_h, new_w = calculate_rectangle_image(
                image, self.scales[process_index['scales']])
        image = resize_numpy_image(image, new_h, new_w)

        # 1st preprocess
        _, h, w = image.shape
        if h > w:
            left, right = 0, w
            start_point = process_index['1st_preprocess'] * \
                    ((h - w) // (self.process_dict['1st_preprocess'] - 1))
            top = start_point
            bottom = top + w
        else:
            top, bottom = 0, h
            start_point = process_index['1st_preprocess'] * \
                    ((w - h) // (self.process_dict['1st_preprocess'] - 1))
            left = start_point
            right = left + h
        image = image[:, top:bottom, left:right]
        _, h, w = image.shape

        assert h == w, "Image shape is not square: {}". format(image.shape)

        # 2nd preproces
        if process_index['2nd_preprocess'] == 0: # top, left
            image = image[:, 0:crop_size, 0:crop_size]
        elif process_index['2nd_preprocess'] == 1: # bottom, left
            image = image[:, h-crop_size:h, 0:crop_size]
        elif process_index['2nd_preprocess'] == 2: # bottom, right
            image = image[:, h-crop_size:h, w-crop_size:w]
        elif process_index['2nd_preprocess'] == 3: # top, right
            image = image[:, 0:crop_size, w-crop_size:w]
        elif process_index['2nd_preprocess'] == 4: # center
            start_point = (h - crop_size) // 2
            image = image[:,
                    start_point:start_point+crop_size,
                    start_point:start_point+crop_size]
        else: # resize
            image = resize_numpy_image(image, crop_size, crop_size)

        _, h, w = image.shape
        assert h == crop_size and w == crop_size, \
                "Image size for the model is incorrect: {}x{}". format(h, w)

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

        return image

    def get_example(self, i):
        process_index = self.index_extractor(i)
        image, label = self.base[process_index['image']]
        image = self.image_preprocess(image, process_index)
        return image, label


class GooglenetTrainDataset(DatasetMixin):

    def __init__(self, path, root, mean, crop_size=224):
        self.base = LabeledImageDataset(path, root)
        self.mean = mean # BGR
        self.crop_size = crop_size
        self.scales = scales
        self.process_dict = {
                '1st_preprocess': 3, # left, center, right cropping
                '2nd_preprocess': 5, # 4-corner and center cropping, resize
                'mirror': 2,
                'scales': len(scales)}
        self.index_extractor = IndexExtractor(self.process_dict)
        self.num_crop = len(self.index_extractor)


    def get_example(self, i):
        crop_size = self.crop_size
        image, label = self.base[i]

        # rectangle resize
        new_h, new_w = calculate_rectangle_image(image, 256)
        image = resize_numpy_image(image, new_h, new_w)

        # random cropping
        top = randint(0, h - crop_size - 1)
        bottom = top + crop_size
        left = randint(0, w - crop_size - 1)
        right = left + crop_size
        image = image[:, top:bottom, left:right]

        # horizontal flopping
        if randint(0, 2):
            image = image[:, :, ::-1]

        # subtract mean
        image = image.transpose((1, 2, 0))
        image -= self.mean.astype(numpy.float32)
        image = image.transpose((2, 0, 1))

        return image, label


class GooglenetMultiLabeledTestDataset(GooglenetTestDataset):

    def __init__(self, path, root, mean, crop_size=224,
            scales=[256, 288, 320, 352]):
        self.base = MultiLabeledImageDataset(path, root)
        self.mean = mean # BGR
        self.crop_size = crop_size
        self.scales = scales
        self.process_dict = {
                '1st_preprocess': 3, # left, center, right cropping
                '2nd_preprocess': 6, # 4-corner and center cropping, resize
                'mirror': 2,
                'scales': len(scales)}
        self.index_extractor = IndexExtractor(self.process_dict)


from chainer.datasets import LabeledImageDataset
from numpy.random import randint

from dataset.Dataset import *
from dataset.multi_labeled_image_dataset import MultiLabeledImageDataset

class ResnetMultiLabeledTestDataset(DatasetMixin):

    def __init__(self, pair, root, mean, crop_size):
        self.mean = mean
        self.scales = [224, 256, 384, 480, 640]
        self.process_index = {
                'crop': 5,
                'mirror': 2,
                'scales': len(self.scales) }
        self.index_extractor = IndexExtractor(
                self.process_index)
        self.base = MultiLabeledImageDataset(pair, root)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.base) * self.num_crop

    def num_crop(self):
        return len(self.index_extractor)


    def preprocess_image(self, image, index):

        # convert gray-scale to RGB-color
        image = convert_grayscale_rgb(image)

        # resize
        scale = self.scales[process_index['scales']]
        new_h, new_w = calculate_rectangle_image(image, scale)
        image = resize_numpy_image(image, new_h, new_w)
        _, h, w = image.shape

        # center cropping
        image = center_crop(image)

        # subtract mean
        mean = resize_numpy_image(mean, scale, scale)
        image -= mean # mean is RGB

        # cropping
        _, h, w = image.shape
        if index['crop'] == 0: # top-left
            top, left = 0, 0
        elif index['crop'] == 1: # bottom-left
            top, left = h - crop_size, 0
        elif index['crop'] == 2: # bottom-right
            top, left = h - crop_size, w - crop_size
        elif index['crop'] == 3: # top-right
            top, left = 0, w - crop_size
        else: # center
            top, left = (h - crop_size) // 2, (w - crop_size) // 2
        bottom, right = top + crop_size, left + crop_size
        image = image[:, top:bottom, left:right]

        # subtract mean
        image = image.transpose((1, 2, 0))
        image -= self.mean.astype(numpy.float32)
        image = image.transpose((2, 0, 1))

        # channel_swap(RGB -> BGR)
        image = channel_swap(image, (2, 1, 0))

        # mirror
        if index['mirror']:
            image = image[:, :, ::-1]

        return image

    def get_example(self, i):
        extracted_index = self.index_extractor(i)
        image, label = self.base[extracted_index['image']]
        image = preprocess_image(image, extracted_index)
        return image, label

class ResnetTrainDataset(DatasetMixin):

    def __init__(self, pair, root, mean, crop_size, random=True):
        self.mean = mean
        self.scales = (256, 480)
        self.base = LabeledImageDataset(pair, root)
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)


    def get_example(self, i):
        image, label = self.base[i]

        # convert gray-scale to RGB-color
        image = convert_grayscale_rgb(image)
        crop_size = self.crop_size

        # resize(random)
        scale = randint(*self.scales)
        new_h, new_w = calculate_rectangle_image(image, scale)
        image = resize_numpy_image(image, new_h, new_w)
        _, h, w = image.shape

        # cropping(random)
        if random:
            top = randint(0, h - crop_size)
            left = randint(0, w - crop_size)
        else: # center croping
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size
        image = image[:, top:bottom, left:right]

        # subtract mean
        image = image.transpose((1, 2, 0))
        image -= self.mean.astype(numpy.float32)
        image = image.transpose((2, 0, 1))

        # channel_swap(RGB -> BGR)
        image = channel_swap(image, (2, 1, 0))

        # mirror(random)
        if self.random and randint(0, 2):
            image = image[:, :, ::-1]

        return image, label

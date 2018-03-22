
import chainer
import numpy as np

class AlexNetValidationOrTestDataset(chainer.dataset.DatasetMinxin):

    num_crop_images = 10

    def __init__(self, path, root, mean, crop_size):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        images = np.ndarray((10, 3, crop_size, crop_size), dtype=image.dtype)

        # top-left and its mirror
        top, left, bottom, right = 0, 0, crop_size, crop_size
        images[0] = self._croped_image(image, top, left, bottom, right)
        images[1] = self._mirror_image(images[0])

        # top-right and its mirror
        top, left, bottom, right = 0, w - crop_size, crop_size, w
        images[2] = self._croped_image(image, top, left, bottom, right)
        images[3] = self._mirror_image(images[2])

        # bottom-right and its mirror
        top, left, bottom, right = h - crop_size, w - crop_size, h, w
        images[4] = self._croped_image(image, top, left, bottom, right)
        images[5] = self._mirror_image(images[4])

        # bottom-left and its mirror
        top, left, bottom, right = h - crop_size, 0, h, crop_size
        images[6] = self._croped_image(image, top, left, bottom, right)
        images[7] = self._mirror_image(images[6])

        # center and its mirror
        top, left = (h - crop_size) // 2, (w - crop_size) // 2
        bottom, right = top + crop_ize, left + crop_size
        images[8] = self._croped_image(image, top, left, bottom, right)
        images[9] = self._mirror_image(images[8])

        return images, np.broadcast_to(label, (self.num_crop_images))

    def _croped_image(self, image, top, left, bottom, right):
        return image[:, top:bottom, left:right] - self.mean[:, top:bottom, left:right]

    def _mirror_image(self, image):
        return image[:, :, ::-1]


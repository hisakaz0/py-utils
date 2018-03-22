#!/usr/bin/env python

from PIL import Image

def resize_image(path, size=256):
    """resize image to 256x256.

    the way to reisize is followed the paper
    'ImageNet Classification with Deep Convolutional
    Neural Networks'

    1. Keeping aspect ratio of image,
       apply scaling like short side to 256.
    2. Cropping in center.
    """

    _portlate  = 0,
    _landscape = 1,

    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w, h = img.size
    if (w == size) and (h == size):
        return img
    shape = _landscape if w > h else _portlate
    shorter_side = w if w < h else h
    longer_side  = w if w > h else h
    magnification = size / shorter_side
    longer_side  *= magnification
    shorter_side *= magnification
    if shape == _landscape:
        resize = (round(longer_side), round(shorter_side))
    else:
        resize = (round(shorter_side), round(longer_side))
    assert round(shorter_side) == size, "this function has bugs"
    scaled_img = img.resize(resize, Image.LANCZOS)
    r_w, r_h = resize
    if (r_h == size) and (r_w == size):
        return scaled_img
    if shape == _landscape:
        left = (r_w - size) // 2
        top = 0
    else:
        left = 0
        top = (r_h - size) // 2
    bottom = top + size
    right = left + size
    box = (left, top, right, bottom)
    cropped_img = scaled_img.crop(box)
    assert (size, size) == cropped_img.size, \
            "this function has bugs"
    return cropped_img


def save_image(img, path, img_format=None):
    if img_format == None:
        img_format = img.format
    img.save(path, img_format)


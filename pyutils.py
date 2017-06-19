#!/usr/bin/env python

import math
import os
import numpy as np

def maximum_rectangle(l):
    if (l==0):
        return 0
    lim = math.ceil(math.sqrt(l)) + 1
    min_leng, min_a, min_b= l+1, 1, l
    for a in range(1, lim):
        if (l%a != 0):
            continue
        b = l//a
        leng = a + b
        if (min_leng > leng):
            min_leng = leng
            min_a, min_b = a, b
    return min_a, min_b


def ensure_dir(path):
    directory = os.path.dirname(path)
    if directory == '':
        return
    if not os.path.exists(directory):
        os.makedirs(directory)

def normalize(x, x_max, x_min):
    return (x - x_min) / (x_max - x_min)


def untiled_image(square, shape):
    """
        square.shape is (a, a)
        shape is (n, m, m)
    """
    cube = np.ndarray(shape)

    start, end = 0, square.shape[0]
    intval, k = shape[1], 0
    for i_s in range(start, end, intval):
        i_e = i_s + intval
        for j_s in range(start, end, intval):
            j_e = j_s + intval
            cube[k] = square[i_s:i_e].T[j_s:j_e].T
            k += 1
    return cube

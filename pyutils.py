#!/usr/bin/env python

import math
import os

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


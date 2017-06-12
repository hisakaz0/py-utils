#!/usr/bin/env python

import math

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


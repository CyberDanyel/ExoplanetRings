# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:46:48 2023

@author: victo
"""

import numpy as np
# exoring_functions

def integrate2d(func, bounds, n):
    width = int(np.sqrt(n))
    xs = np.broadcast_to(np.linspace(bounds[0][0], bounds[0][1], width), (width, width))
    ys = np.broadcast_to(np.array([np.linspace(bounds[1][0], bounds[1][1], width)]).T, (width, width))
    area = (xs[0][1] - xs[0][0]) * (ys[1][0] - ys[0][0])
    arr = func(xs, ys)
    return np.sum(arr)*area
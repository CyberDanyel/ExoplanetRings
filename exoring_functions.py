# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:46:48 2023

@author: victo
"""

import numpy as np
# exoring_functions

def integrate2d(func, bounds, n):
    '''
    2D integration by basic Riemann sum

    Parameters
    ----------
    func : function object
        The function to be integrated, with two input variables
    bounds : iterable w. shape 2,2
        A tuple of floats specifying the bounds of the respective variables
    n : int
        The number of subdivisions for each variable, total number of samples is n^2.

    Returns
    -------
    float
        The definite integral

    '''
    xs = np.broadcast_to(np.linspace(bounds[0][0], bounds[0][1], n), (n, n))
    ys = np.broadcast_to(np.array([np.linspace(bounds[1][0], bounds[1][1], n)]).T, (n, n))
    area = (xs[0][1] - xs[0][0]) * (ys[1][0] - ys[0][0])
    arr = func(xs, ys)
    return np.sum(arr)*area

def integrate2d_conv(func, bounds, tol, max_n):
    width = 10
    total_area = (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0])
    old_val = func(np.mean(bounds[0]), np.mean(bounds[1])) * total_area
    while width < max_n-2:
        new_val = integrate2d(func, bounds, width)
        err = new_val-old_val
        if np.abs(err) < tol:
            return new_val
        else:
            old_val = new_val
            width=int(width*1.1)
    print('Integration Warning:Max n reached, delta is %.3e'%err)
    return new_val
        
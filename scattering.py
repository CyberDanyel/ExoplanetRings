# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:58:43 2023

@author: victo
"""

# scattering

import numpy as np
import scipy.special as spe


def rayleigh(alpha, w):
    # theta = np.pi - alpha for the specific case of Rayleigh scattering forward and backward scattering are
    # symmetric - irrelevant if we use theta or alpha
    return w * (1 / (3 * np.pi)) * (1 + np.cos(alpha) ** 2)

# def mie(theta, a, wavelength):
#    l = wavelength

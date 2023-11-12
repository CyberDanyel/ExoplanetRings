# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:30:24 2023

@author: victo
"""

# spectra

import numpy as np
import pandas as pd
import scipy.interpolate as spint
import scipy.integrate as spi

class Material:
    def __init__(self, filename):
        data = pd.read_csv(filename)
        wavelengths = np.array(data['Wavelength'])
        #r_data = np.array(data['Reflectivity'])
        #a_data = np.array(data['Absorption'])
        #t_data = np.array(data['Transmittivity'])
        self.n = np.array(data['Refractive Index (real)'])
        self.k = np.array(data['Refractive Index (imaginary)'])
        self.m = self.n + self.k * (0+1j)
        #self.reflectivity = spint.CubicSpline(wavelengths, r_data)
        #self.absorption = spint.CubicSpline(wavelengths, a_data)
        #self.transmittivity = spint.CubicSpline(wavelengths, t_data)
        
    #def R(self, I, bandpass):
    #    return I*spi.quad(self.reflectivity, bandpass[0], bandpass[1])[0]/(bandpass[1]-bandpass[0])
    
    #def T(self, I, bandpass):
    #    return I*spi.quad(self.transmittivity, bandpass[0], bandpass[1])[0]/(bandpass[1]-bandpass[0])
    
    #def A(self, I, bandpass):
    #    return I*spi.quad(self.absorption, bandpass[0], bandpass[1])[0]/(bandpass[1]-bandpass[0])
    
        
        
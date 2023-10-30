# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:58:43 2023

@author: victo
"""

# scattering

import numpy as np
import scipy.special as spe
import scipy.interpolate as spip
import scipy.integrate as spi


class ScatteringLaw:
    def __init__(self, albedo, func):
        self.albedo = albedo
        self.func = func
        self.norm = spi.quad(func, 0, np.pi)[0]
        
    def __call__(self, alpha):
        return (self.albedo/self.norm) * self.func(np.pi-np.abs(alpha))

class Lambert(ScatteringLaw):
    def __init__(self, albedo):
        ScatteringLaw.__init__(self, albedo, lambda x:1.)
        
class Rayleigh(ScatteringLaw):
    def __init__(self, albedo):
        ScatteringLaw.__init__(self, albedo, self.rayleigh_func)
        
    def rayleigh_func(self, theta):
        return (1+np.cos(theta)**2)
       
class HG(ScatteringLaw):
    #henyey-greenstein
    def __init__(self, g, albedo):
        self.g = g
        ScatteringLaw.__init__(self, albedo, self.hg_func)
    
    def hg_func(self, theta):
        return (2*(1 - self.g**2)) / (1 + self.g**2 - 2*self.g*np.cos(theta))**(1.5)
        
class Empirical(ScatteringLaw):
    def __init__(self, filename, albedo):
        self.filename = filename
        self.points = np.loadtxt(filename, delimiter = ',')
        emp_func = spip.CubicSpline(self.points[0], self.points[1])
        ScatteringLaw.__init__(self, albedo, emp_func)

class Jupiter(ScatteringLaw):
    #from Dyudina et al. 2005
    def __init__(self, albedo):
        self.g1 = 0.8
        self.g2 = -.38
        self.f = 0.9
        ScatteringLaw.__init__(self, albedo, self.jupiter_func)
    
    def hg_func(self, g, theta):
        return (2*(1 - g**2)) / (1 + g**2 - 2*g*np.cos(theta))**(1.5)
        
    def jupiter_func(self, theta):
        return self.f*self.hg_func(self.g1, theta) + (1-self.f)*self.hg_func(self.g2, theta)
    

#todo: mie scattering
# def mie(theta, a, wavelength):
#    l = wavelength

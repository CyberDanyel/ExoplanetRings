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
import PyMieScatt as msc

def lambert_phase_func(alpha):
    #important enough to make its own thing
    return 2/3 * ((np.pi-np.abs(alpha))*np.cos(np.abs(alpha))+np.sin(np.abs(alpha)))
    

class SingleScatteringLaw:
    def __init__(self, albedo, func):
        self.albedo = albedo
        self.func = func
        self.norm = spi.quad(func, 0, np.pi)[0]
    def __call__(self, alpha):
        return (self.albedo/self.norm) * self.func(np.pi-np.abs(alpha))

class Lambert(SingleScatteringLaw):
    def __init__(self, albedo):
        SingleScatteringLaw.__init__(self, albedo, lambda x:1.)
        
class Rayleigh(SingleScatteringLaw):
    def __init__(self, albedo):
        SingleScatteringLaw.__init__(self, albedo, self.rayleigh_func)
        
    def rayleigh_func(self, theta):
        return (1+np.cos(theta)**2)
       
class HG(SingleScatteringLaw):
    #henyey-greenstein
    def __init__(self, g, albedo):
        self.g = g
        SingleScatteringLaw.__init__(self, albedo, self.hg_func)
    
    def hg_func(self, theta):
        return (2*(1 - self.g**2)) / (1 + self.g**2 - 2*self.g*np.cos(theta))**(1.5)
        
class SingleEmpirical(SingleScatteringLaw):
    def __init__(self, filename, albedo):
        self.filename = filename
        self.points = np.loadtxt(filename, delimiter = ',')
        emp_func = spip.CubicSpline(self.points[0], self.points[1])
        SingleScatteringLaw.__init__(self, albedo, emp_func)



class SingleMie(SingleScatteringLaw):
    def __init__(self, albedo, X, m):
        self.X = X
        self.m = m
        SingleScatteringLaw.__init__(self, albedo, np.vectorize(self.mie_func))
        
    def mie_func(self, theta):
        S1, S2 = msc.MieS1S2(self.m, self.X, np.cos(theta))
        return np.abs(S1)**2 + np.abs(S2)**2

#general functions independent of specific scattering situation
def psi(x, n):
    return x*spe.spherical_jn(n, x)
def zeta(x, n):
    return np.sqrt((np.pi * x)/2) * spe.hankel2(n+0.5, x)
def psi_prime(x, n):
    dx = 1e-8
    return (psi(x+dx, n) - psi(x, n))/dx
def zeta_prime(x, n):
    dx = 1e-8
    return (zeta(x+dx, n) - zeta(x, n))/dx


class Jupiter(SingleScatteringLaw):
    #from Dyudina et al. 2005
    def __init__(self, albedo):
        self.g1 = 0.8
        self.g2 = -.38
        self.f = 0.9
        SingleScatteringLaw.__init__(self, albedo, self.jupiter_func)
    
    def hg_func(self, g, theta):
        return (2*(1 - g**2)) / (1 + g**2 - 2*g*np.cos(theta))**(1.5)
        
    def jupiter_func(self, theta):
        return self.f*self.hg_func(self.g1, theta) + (1-self.f)*self.hg_func(self.g2, theta)
    


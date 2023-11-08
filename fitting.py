# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:37:10 2023

@author: victo
"""

#fitting

import numpy as np
import matplotlib.pyplot as plt
import exoring_objects
import scattering
import scipy.optimize as op

AU = 1.495978707e13
L_SUN = 3.828e33
R_JUP = 6.9911e9
R_SUN = 6.957e10
AU_TO_JUP = AU / R_JUP
SUN_TO_JUP = R_SUN / R_JUP
SUN_TO_AU = AU / R_SUN


star = exoring_objects.Star(1, SUN_TO_JUP, 0.1*AU_TO_JUP, 1)

def gaussian(x, mu, sigma):
    return (1/np.sqrt(2*np.pi*sigma))*np.exp(-0.5*((x-mu)/(sigma))**2)
    
def log_likelihood_ring(data, planet_sc_name, ring_sc_name, star, *params):
    alpha = data[0]
    I = data[1]
    I_errs = data[2]
    model_ringed_planet = FittingRingedPlanet(planet_sc_name, ring_sc_name, star, *params)
    x = model_ringed_planet.light_curve(alpha)
    return -np.sum(np.log(gaussian(x, I, I_errs)))

def log_likelihood_planet(data, sc_name, star, *params):
    alpha = data[0]
    I = data[1]
    I_errs = data[2]
    model_planet = FittingPlanet(sc_name, star, *params)
    x = model_planet.light_curve(alpha)
    return -np.sum(np.log(gaussian(x, I, I_errs)))
    
def minimize_logL_ring(data, planet_sc_name, ring_sc_name, star, p0, bounds=None):
    m = op.minimize(lambda p:log_likelihood_ring(data, planet_sc_name, ring_sc_name, star, *p), p0, bounds=bounds)
    return m.x

def minimize_logL_planet(data, sc_name, star, p0, bounds=None):
    m = op.minimize(lambda p:log_likelihood_planet(data, sc_name, star, *p), p0, bounds=bounds)
    return m.x

class FittingPlanet(exoring_objects.Planet):
    def __init__(self, sc_law_name, star, *params):
        if sc_law_name == 'Lambert':
            albedo, radius = params[:2]
            sc_law = scattering.Lambert(albedo)
        elif sc_law_name == 'Rayleigh':
            albedo, radius = params[:2]
            sc_law = scattering.Rayleigh(albedo)
        elif sc_law_name == 'HG':
            albedo, g, radius = params[:3]
            sc_law = scattering.HG(g, albedo)
        elif sc_law_name == 'Jupiter':
            albedo, radius = params[:2]
            sc_law = scattering.Jupiter(albedo)
        else:
            raise TypeError('Ain\'t no law with that name round these parts, pardner')
            
        exoring_objects.Planet.__init__(self, sc_law, radius, star)
    
class FittingRingedPlanet(exoring_objects.RingedPlanet, FittingPlanet):
    def __init__(self, planet_sc_name, ring_sc_name, star, *params):
        
        FittingPlanet.__init__(self, planet_sc_name, *params)
        if ring_sc_name == 'Lambert':
            albedo, inner_rad, ring_width, n_x, n_y, n_z = params[-6:]
            ring_sc = scattering.Lambert(albedo)
        elif ring_sc_name == 'Rayleigh':
            albedo, inner_rad, ring_width, n_x, n_y, n_z = params[-6:]
            ring_sc = scattering.Rayleigh(albedo)
        elif ring_sc_name == 'HG':
            albedo, g, inner_rad, ring_width, n_x, n_y, n_z = params[-7:]
            ring_sc = scattering.HG(g, albedo)
        elif ring_sc_name == 'Mie':
            albedo, X, m, inner_rad, ring_width, n_x, n_y, n_z = params[-8:]
            ring_sc = scattering.Mie(albedo, X, m)
        else:
            raise TypeError('Ain\'t no law with that name round these parts, pardner')
         
            
        exoring_objects.RingedPlanet.__init__(self, self.sc_law, self.radius, ring_sc, inner_rad, inner_rad + ring_width, [n_x, n_y, n_z], star)
        

def generate_data(test_planet):
    test_alphas = list(np.linspace(-np.pi, -.3, 10)) + list(np.linspace(-.25, .25, 10)) + list(np.linspace(.3, np.pi, 10))
    test_alphas = np.array(test_alphas)
    I = test_planet.light_curve(test_alphas)
    errs = 0.02*I + 1e-8
    noise_vals = np.random.normal(size = len(test_alphas))
    data_vals = errs*noise_vals + I
    data = np.array([test_alphas, data_vals, errs])
    return data
    

test_planet = exoring_objects.Planet(scattering.Jupiter(1), 1, star)  
data = generate_data(test_planet)


planet_vals = minimize_logL_planet(data, 'Rayleigh', star, np.array([1, 1]), bounds = [(0, np.inf),(0., np.inf)])
ring_vals = minimize_logL_ring(data, 'Jupiter', 'Rayleigh', star, np.array([.2, .5, 1., 1., 1., 1., 1., 1.]), bounds = [(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0.01, np.inf), (0.01, np.inf), (0.01, np.inf)])

alphas = np.linspace(-np.pi, np.pi, 10000)
plt.style.use('barbie')
plt.title('This Barbie is NOT a ringed planet')
plt.errorbar(data[0], data[1], data[2], fmt='.')
plt.plot(alphas, FittingPlanet('Rayleigh', star, *planet_vals).light_curve(alphas), label='Planet')
plt.plot(alphas, FittingRingedPlanet('Jupiter', 'Rayleigh', star, *ring_vals).light_curve(alphas), label='Planet+Ring')

plt.legend()
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

def fit_func_ring(alphas, r_planet, planet_albedo, r_inner, r_outer, ring_angle, ring_albedo):
    sc_law_planet = scattering.Lambert(planet_albedo)
    model_planet = exoring_objects.Planet(sc_law_planet, r_planet, star)
    ring_normal = [np.cos(ring_angle), np.sin(ring_angle), 0]
    sc_law_ring = scattering.Rayleigh(ring_albedo)
    model_ring = exoring_objects.Ring(sc_law_ring, r_inner, r_outer, ring_normal, star)
    return model_planet.light_curve(alphas) + model_ring.light_curve(alphas)

def fit_func_planet(alphas, r, albedo):
    sc_law = scattering.Lambert(albedo)
    model_planet = exoring_objects.Planet(sc_law, r, star)
    return model_planet.light_curve(alphas)
    
test_planet = exoring_objects.Planet(scattering.Lambert(1), 1, star)
test_ring = exoring_objects.Ring(scattering.Rayleigh(1), 2, 3, [1, 0.2, 0.1], star)

alphas = np.linspace(-np.pi, np.pi, 10000)

data = test_planet.light_curve(alphas) + test_ring.light_curve(alphas)

ring_vals, ring_cov = op.curve_fit(fit_func_ring, alphas, data, p0 = [1, 0.8, 1, 2, 0.2, 1])
planet_vals, planet_cov = op.curve_fit(fit_func_planet, alphas, data, p0 = [2, 0.8])

plt.style.use('the_usual')

plt.plot(alphas, data, label='Data')
plt.plot(alphas, fit_func_ring(alphas, *ring_vals), label='Fitted planet with ring')
plt.plot(alphas, fit_func_planet(alphas, *planet_vals), label = 'Fitted planet')

plt.legend()





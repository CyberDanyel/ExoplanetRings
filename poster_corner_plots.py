# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:41:44 2024

@author: victo
"""

#poster corner plots

import numpy as np
import exoring_objects
import scattering
import materials
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import threading
import concurrent.futures

with open('constants.json') as json_file:
    constants = json.load(json_file)

R_JUP = constants['R_JUP']
M_JUP = constants['M_JUP']
R_SUN = constants['R_SUN']
AU = constants['AU']


plt.style.use('poster')
bandpass = (11e-6, 14e-6)
star = exoring_objects.Star(4800, R_SUN, .1*AU, 1.)

silicate = materials.RingMaterial('materials/silicate_small.inp', 361, 500)
atmos = materials.Atmosphere(scattering.Jupiter,  [M_JUP, R_JUP], star)
sc_sil = scattering.WavelengthDependentScattering(silicate, bandpass, star.planck_function)
sc_atmos = scattering.WavelengthDependentScattering(atmos, bandpass, star.planck_function)

def gaussian(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)

def generate_data():
    alphas = np.linspace(-np.pi, np.pi, 100)
    ring_params = [1.2*R_JUP, 1.5*R_JUP, [1., 0.5, 0.1], star]
    synthetic_planet = exoring_objects.RingedPlanet(sc_atmos, 1*R_JUP, sc_sil, *ring_params)
    synthetic_curve = synthetic_planet.light_curve(alphas)/star.luminosity
    errors = np.random.normal(size = np.size(alphas))
    errors *= 0.2e-6
    synthetic_curve += errors
    synthetic_curve[synthetic_curve < 0] *= 0
    return alphas, synthetic_curve

ring_params = [1.2*R_JUP, 2*R_JUP, [1., 0.5, 0.1], star]
synthetic_planet = exoring_objects.RingedPlanet(sc_atmos, 1*R_JUP, sc_sil, *ring_params)

def log_likelihood(data, params):
    planet_r, ring_gap, ring_width, ring_normal_phi, ring_normal_theta = params
    alphas, data_values = data
    model_ring_params = [planet_r + ring_gap, planet_r + ring_gap + ring_width, [np.sin(ring_normal_theta)*np.cos(ring_normal_phi), np.sin(ring_normal_theta)*np.sin(ring_normal_phi), np.cos(ring_normal_theta)]]
    model_planet = exoring_objects.RingedPlanet(sc_atmos, planet_r, sc_sil, *model_ring_params, star)
    model_values = model_planet.light_curve(alphas)/star.luminosity
    probabilities = gaussian(model_values, data_values, .2e-6)
    log_L = np.sum(np.log(probabilities))
    return log_L

ringn_thetas = np.linspace(0, np.pi/2, 10)
ringn_phis = np.linspace(-np.pi/2, np.pi/2, 10)
planet_rs = np.linspace(0, 2*R_JUP, 10)
ring_gaps = np.linspace(0, 1*R_JUP, 10)
ring_widths = np.linspace(0, 2*R_JUP, 10)

log_space = np.zeros((len(ringn_thetas), len(ringn_phis), len(planet_rs), len(ring_gaps), len(ring_widths)))

data = generate_data()
#plt.errorbar(data[0], data[1], .2e-6, fmt = '.')
#plt.plot(data[0], synthetic_planet.light_curve(data[0])/star.luminosity)


index_array = np.indices(np.shape(log_space))
coord_iterable = np.reshape(index_array, (5, np.size(log_space))).T
def find_logL_in_space(i):
    a, b, c, d, e = coord_iterable[i]
    logL = log_likelihood(data, (planet_rs[c], ring_gaps[d], ring_widths[e], ringn_phis[b], ringn_thetas[a]))
    log_space[a, b, c, d, e] += logL
    #return logL

def find_logL_iterable(it):
    for i in it:
        find_logL_in_space(i)

if __name__ == '__main__':
    with Pool(4) as p:
        log_list = list(tqdm(p.imap(find_logL_in_space, range(len(coord_iterable)), chunksize = 10), total = len(coord_iterable), desc = 'Running Models'))
   
#log_space = np.reshape(np.array(log_list), np.shape(log_space))
reduce_thetas = np.zeros(np.shape(log_space)[1:])
for i, row in enumerate(np.exp(log_space)):
    reduce_thetas += np.sin(ringn_thetas[i]) * row

reduce_thetas *= (ringn_thetas[1] - ringn_thetas[0])
reduce_phis = np.sum(reduce_thetas, axis = 0) * (ringn_phis[1] - ringn_phis[0])

def make_corner_plots(angle_integrated_data):
    fig, axes = plt.subplots(3, 3)
    axes[0,1].axis('off')
    axes[0,2].axis('off')
    axes[1,2].axis('off')
    planetR_integrated = np.sum(angle_integrated_data, axis = 0) * (planet_rs[1] - planet_rs[0])
    axes[1,1].contourf(ring_gaps/R_JUP, ring_widths/R_JUP, planetR_integrated)
    ring_gaps_integrated = np.sum(angle_integrated_data, axis = 1) * (ring_gaps[1] - ring_gaps[0])
    axes[2,0].contourf(planet_rs/R_JUP, ring_widths/R_JUP, ring_gaps_integrated)
    ring_widths_integrated = np.sum(angle_integrated_data, axis = 2) * (ring_widths[1] - ring_widths[0])
    axes[1,0].contourf(planet_rs/R_JUP, ring_gaps/R_JUP, ring_widths_integrated)
    ring_width_dist = np.sum(planetR_integrated, axis = 0) * (ring_gaps[1] - ring_gaps[0])
    axes[2,2].plot(ring_widths/R_JUP, ring_width_dist)
    ring_gaps_dist = np.sum(planetR_integrated, axis = 1) * (ring_widths[1] - ring_gaps[0])
    axes[1,1].plot(ring_gaps/R_JUP, ring_gaps_dist)
    
    
    

make_corner_plots(reduce_phis)
plt.show()
plt.savefig('attempt.pdf')
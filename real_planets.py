# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:56:23 2024

@author: victo
"""

# REAL planets

import numpy as np
import matplotlib.pyplot as plt

import json
import exoring_objects
import scattering
import materials
#import fitting

with open('constants.json') as json_file:
    constants = json.load(json_file)

plt.style.use('the_usual')
bandpass = (11.43e-6, 14.17e-6) # f1280w
R_SUN = constants['R_SUN']
M_SUN = constants['M_SUN']
AU = constants['AU']
R_JUP = constants['R_JUP']
R_EARTH = constants['R_EARTH']
M_EARTH = constants['M_EARTH']
RHO_JUP = constants['RHO_JUP']
alphas = np.linspace(-np.pi, np.pi, 10000)
phis = [np.pi/16, np.pi/8, 3*np.pi/16, np.pi/4, 5*np.pi/16]
phi_labels = [r'$\frac{\pi}{16}$', r'$\frac{\pi}{8}$', r'$\frac{3\pi}{16}$', r'$\frac{\pi}{4}$', r'$\frac{5\pi}{16}$']

silicate = materials.RingMaterial('materials/silicate_small.inp', 361, 500)

def run_real_planet(filename):
    with open(filename) as planet_file:
        vals = json.load(planet_file)

        # defining constants in the system
        T_star = vals['T_star']
        R_star = vals['R_star']*R_SUN
        distance = vals['dist']*AU
        M_star = vals['M_star']*M_SUN
        M = vals['M']*M_EARTH
        R = vals['R']*R_JUP
        density = 3*M/(4*np.pi*(R**3))
        R_reduced = R*(density/RHO_JUP)**(1/3)
        saturn_J2 = 0.01656

        R_hill = distance * (M / (3*M_star))**(1/3)
        R_roche = R_reduced * 2.45 * (1/3)**(1/3) # the density of silicate is about 3 times that of Jupiter
        R_laplace = (2*saturn_J2*R_reduced**2*distance**3*(M/M_star))**(1/5)
        R_max = np.array([R_hill, R_roche, R_laplace])
        max_phis = np.arccos((R/R_max)**2)

        # defining objects
        star = exoring_objects.Star(T_star, R_star, distance, M_star)
        atmos = materials.Atmosphere(scattering.Jupiter, [M, R], star)
        atmos_sc = scattering.WavelengthDependentScattering(atmos, bandpass, star.planck_function)
        ring_sc = scattering.WavelengthDependentScattering(silicate, bandpass, star.planck_function)

        # calculate light curves
        unringed_planet = exoring_objects.Planet(atmos_sc, R, star)
        unringed_light_curve = unringed_planet.light_curve(alphas)
        ringed_light_curves = np.zeros((len(phis), len(alphas)))
        max_light_curves = np.zeros((len(max_phis), len(alphas)))
        for n, phi in enumerate(phis):
            normal = [np.cos(phi), np.sin(phi), 0]
            ringed_planet = exoring_objects.RingedPlanet(atmos_sc, R_reduced, ring_sc, R_reduced, R/np.sqrt(np.cos(phi)), normal, star)
            ringed_light_curves[n] += ringed_planet.light_curve(alphas)
        for n, phi in enumerate(max_phis):
            normal = [np.cos(phi), np.sin(phi), 0]
            ringed_planet = exoring_objects.RingedPlanet(atmos_sc, R_reduced, ring_sc, R_reduced, R / np.sqrt(np.cos(phi)), normal, star)
            max_light_curves[n] += ringed_planet.light_curve(alphas)

    return unringed_light_curve/star.luminosity, ringed_light_curves/star.luminosity, max_light_curves/star.luminosity

def real_planet_light_curve_plots(planetname):
    unringed_light_curve, ringed_light_curves, max_light_curves = run_real_planet('exoplanet_info/' + planetname + '.json')
    fig, ax = plt.subplots(1, 1, figsize = [10, 5])
    for n, ringed_light_curve in enumerate(ringed_light_curves):
        plt.plot(alphas, ringed_light_curve, label = r'Ringed planet, $\varphi=$' + phi_labels[n], color = [n/len(ringed_light_curves), 0, 0])
    plt.plot(alphas, unringed_light_curve, label='Ringless, inflated planet', color='xkcd:cornflower blue')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{planetname}.pdf')

def real_planet_asymmetry_plots(planetname):
    unringed_light_curve, ringed_light_curves, max_light_curves = run_real_planet('exoplanet_info/' + planetname + '.json')
    front_half = ringed_light_curves[:,:int(len(alphas)/2)]
    back_half = ringed_light_curves[:,int(len(alphas)/2):]
    residuals = back_half - front_half[:,::-1] # only works for an even number of alphas
    fig, ax = plt.subplots(1, 1, figsize=[6, 4])
    for n, ringed_light_curve in enumerate(ringed_light_curves):
        plt.plot(alphas[int(len(alphas)/2):], residuals[n], label=r'$\varphi=$' + phi_labels[n], color=[n / len(ringed_light_curves), 0, 0])
    plt.ylabel(r'Asymmetry$/L_{\odot}$')
    plt.xlabel(r'Phase angle $\alpha$')
    plt.xlim(0, np.pi)
    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    plt.legend(handlelength = .8)
    plt.tight_layout()
    plt.savefig(f'{planetname}_asymmetry.pdf')

def real_planet_max_plots(planetname):
    unringed_light_curve, ringed_light_curves, max_light_curves = run_real_planet('exoplanet_info/' + planetname + '.json')
    front_half = max_light_curves[:, :int(len(alphas) / 2)]
    back_half = max_light_curves[:, int(len(alphas) / 2):]
    residuals = back_half - front_half[:, ::-1]  # only works for an even number of alphas
    fig, ax = plt.subplots(1, 1, figsize=[6, 4])
    labels = ['Hill', 'Roche', 'Laplace']
    for n, ringed_light_curve in enumerate(max_light_curves):
        plt.plot(alphas[int(len(alphas) / 2):], residuals[n], label=labels[n], color=[n / len(max_light_curves), 0, 0])
    plt.ylabel(r'Asymmetry$/L_{\odot}$')
    plt.xlabel(r'Phase angle $\alpha$')
    plt.xlim(0, np.pi)
    ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    plt.legend(handlelength=.8)
    plt.tight_layout()
    plt.savefig(f'{planetname}_max_asymmetry.pdf')

real_planet_max_plots('kepler87c')

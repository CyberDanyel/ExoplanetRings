# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 23:18:04 2024

@author: victo
"""

# poster scattering plots
import numpy as np
import matplotlib.pyplot as plt

import scattering
import materials
import exoring_objects

plt.style.use('poster')
AU = 1.495978707e11
L_SUN = 3.828e26
R_JUP = 6.9911e7
R_SUN = 6.957e8
JUP_TO_AU = AU / R_JUP
SUN_TO_JUP = R_SUN / R_JUP
AU_TO_SUN = AU / R_SUN
AU_TO_JUP = AU / R_JUP
M_JUP = 1.898e27

plt.style.use('poster')

star = exoring_objects.Star(L_SUN, R_SUN, .1*AU, 1.)
silicate = materials.RingMaterial('materials/silicate_small.inp', 361, 500)
atmos = materials.Atmosphere(scattering.Rayleigh,  [M_JUP, R_JUP], star)

def albedo_plot(ax):
    wavs = np.linspace(min(silicate.wavelengths), 3e-5, 1000)
    ax.plot(wavs*1e6, silicate.albedo(wavs))
    #ax.plot(wavs*1e6, atmos.albedo(wavs))
    ax.set_xlabel(r'Wavelength ($\mathrm{\mu m})$')
    ax.set_ylabel(r'Silicate albedo')
    plt.tight_layout()

def silicate_phase_func_plot(ax):
    thetas = np.linspace(0, np.pi, 1000)
    sc_mie = scattering.Mie(1., 4, 1.5+.1j)
    sc_atmos = scattering.WavelengthDependentScattering(atmos, (1e-5, 1.4e-5), star.planck_function)
    sc_dust = scattering.WavelengthDependentScattering(silicate, (1e-5, 1.4e-5), star.planck_function)
    ax.plot(np.cos(thetas), sc_atmos(np.pi-thetas), label = 'Rayleigh scattering')
    ax.plot(np.cos(thetas), sc_dust(np.pi-thetas), label = 'Mie scattering, averaged across size dist.')
    ax.plot(np.cos(thetas), sc_mie(np.pi-thetas), label = 'Mie scattering, individual particle')
    ax.set_xlabel(r'$\cos\Theta$')
    ax.set_yscale('log')
    ax.set_ylabel('Phase function')
    ax.legend()
    plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (5, 3))
silicate_phase_func_plot(ax)
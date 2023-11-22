# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:43:21 2023

@author: victo
"""

#wavelength testing

import numpy as np
import matplotlib.pyplot as plt

import exoring_objects
import scattering
import materials

AU = 1.495978707e13
L_SUN = 3.828e33
R_JUP = 6.9911e9
R_SUN = 6.957e10
JUP_TO_AU = AU / R_JUP
SUN_TO_JUP = R_SUN / R_JUP
AU_TO_SUN = AU / R_SUN
AU_TO_JUP = AU / R_JUP

bandpass = (3, 5)

material = materials.RingMaterial('materials/saturn_ring.inp', 361, 500)
sc_ring = scattering.WavelengthDependentScattering(material, bandpass)
sc_planet = scattering.Jupiter(0.1)

star = exoring_objects.Star(1, R_SUN, 0.1*AU, 1.)
ringed_planet = exoring_objects.RingedPlanet(sc_planet, R_JUP, sc_ring, 1.5*R_JUP, 3.*R_JUP, [1., 1., 0.1], star)

alphas = list(np.linspace(-np.pi, -.1, 2000)) + list(np.linspace(-.1, .1, 5000)) + list(np.linspace(.1, np.pi, 2000))

light_curve = ringed_planet.light_curve(alphas)

plt.style.use('the_usual')
plt.plot(alphas, light_curve)


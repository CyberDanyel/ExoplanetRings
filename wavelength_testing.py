# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:43:21 2023

@author: victo
"""

#wavelength testing

import numpy as np

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

bandpass = (2, 20)

test_mat = materials.RingMaterial('dustkapscatmat.inp', 361, 300)
test_sc_ring = scattering.WavelengthDependentScattering(test_mat, bandpass)
test_sc_planet = scattering.Jupiter(1.)

star = exoring_objects.Star(L_SUN, R_SUN, 0.1*AU, 1.)
test_ringed_planet = exoring_objects.RingedPlanet(test_sc_planet, R_JUP, test_sc_ring, 1.5*R_JUP, 3.*R_JUP, [1., 1., 0.1], star)

alphas = np.linspace(-np.pi, np.pi, 10000)
vals = test_ringed_planet.light_curve(alphas)
vals *= star.L((bandpass[0]*1e-6, bandpass[1]*1e-6))/star.luminosity



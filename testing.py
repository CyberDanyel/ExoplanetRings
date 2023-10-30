# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:12:36 2023

@author: victo
"""

# testing

import exoring_objects
import scattering
import numpy as np
import matplotlib.pyplot as plt

AU = 1.495978707e13
L_SUN = 3.828e33
R_JUP = 6.9911e9
R_SUN = 6.957e10
JUP_TO_AU = AU / R_JUP
SUN_TO_JUP = R_SUN / R_JUP
SUN_TO_AU = AU / R_SUN

star = exoring_objects.Star(1, SUN_TO_JUP, 0.1 * JUP_TO_AU, 1)

planet = exoring_objects.Planet(1, 1, star)

ring_normal = np.array([0.5, 2., 1.])
ring_normal /= np.sqrt(np.sum(ring_normal**2))
ring = exoring_objects.Ring(1, 1.1, 2., ring_normal, star)

alphas = np.array(list(np.linspace(-np.pi, -.06, 1000)) + list(np.linspace(-.1, .06, 3000)) + list(np.linspace(.06, np.pi, 1000)))

planet_curve = planet.light_curve(alphas)
ring_curve = ring.light_curve(alphas)

plt.style.use('the_usual')
plt.plot(alphas, ring_curve, label = 'Ring')
plt.plot(alphas, planet_curve, label = 'Planet')
plt.plot(alphas, planet_curve + ring_curve, label = 'Ring + Planet')
plt.legend()
plt.show()




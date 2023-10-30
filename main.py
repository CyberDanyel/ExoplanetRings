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
import exoring_functions
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as tck

AU = 1.495978707e13
L_SUN = 3.828e33
R_JUP = 6.9911e9
R_SUN = 6.957e10
JUP_TO_AU = AU / R_JUP
SUN_TO_JUP = R_SUN / R_JUP
SUN_TO_AU = AU / R_SUN

star = exoring_objects.Star(1, SUN_TO_JUP, 0.1 * JUP_TO_AU, 1)

planet = exoring_objects.Planet(1, SUN_TO_JUP*0.5, star)

star.planet = planet

ring_normal = np.array([0.5, 2., 1.])
ring_normal /= np.sqrt(np.sum(ring_normal ** 2))
ring = exoring_objects.Ring(1, 1.1, 2., ring_normal, star)

alphas = np.array(
    list(np.linspace(-np.pi, -.06, 1000)) + list(np.linspace(-.1, .06, 3000)) + list(np.linspace(.06, np.pi, 1000)))

planet_curve = planet.light_curve(alphas)
ring_curve = ring.light_curve(alphas)
star_curve = star.light_curve(alphas)

plt.style.use('the_usual.mplstyle')

fig1, ax1 = exoring_functions.generate_plot_style()
ax1.plot(alphas/np.pi, planet_curve, label='Planet')
ax1.plot(alphas/np.pi, ring_curve, label='Ring')
ax1.plot(alphas/np.pi, planet_curve + ring_curve, label='Ring + Planet')
ax1.legend()
plt.savefig('images/Planet & Rings Light Curve')

fig2, ax2 = exoring_functions.generate_plot_style()
ax2.plot(alphas/np.pi, planet_curve+ring_curve+star_curve, label='Star + Planet + Ring')
ax2.legend()
plt.savefig('images/Full Light Curve')

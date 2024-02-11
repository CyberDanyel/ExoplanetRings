# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 14:26:49 2024

@author: victo
"""

# poster plots

import numpy as np
import matplotlib.pyplot as plt

import exoring_objects
import scattering
import materials
import socket

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
bandpass = (10e-6, 14e-6)
star = exoring_objects.Star(L_SUN, R_SUN, .1*AU, 1.)

fig1, ax1 = plt.subplots(1, 1, figsize = (6, 4))
import poster_phase_curves as phase_curves
phase_curves.make_full_plot(ax1)
#plt.savefig('example_phase_curve.svg', transparent=True)

fig2, ax2 = plt.subplots(1, 1, figsize = (2, 1))
phi_string = r'$\phi_{1, 2} = \frac{R\sin\alpha \pm \sqrt{r_{\star}^2 - r^2 \cos^2\theta}}{\sin\theta}$'
ax2.text(.1, .5, phi_string)
ax2.set_axis_off()
#plt.savefig('phi_bounds.svg', transparent=True)

fig3, ax3 = plt.subplots(1, 1, figsize = (3, 1))
theta_string = r'$\theta_{1, 2} = \frac{\pi}{2} \pm \arccos\left(\frac{r^2 + R^2\sin^2\alpha - r_\star^2}{2rR\sin\alpha} \right)$'
ax3.text(.1, .5, theta_string)
ax3.set_axis_off()
#plt.savefig('theta_bounds.svg', transparent = True)

fig4, ax4 = plt.subplots(1, 1, figsize = (4, 1.5))
int_string = r'$\Psi(\alpha) = \frac{1}{\pi}\int_{0}^{\pi}d\theta\sin\theta\int_{\alpha-\pi/2}^{\pi/2}d\phi\mu \mu_\star \Phi(\mu, \mu_\star)$'
ax4.text(.1, .5, int_string)
ax4.set_axis_off()

fig5, ax5 = plt.subplots(1, 1, figsize = (6, 2.2))
phase_curves.make_example_plot(ax5)
plt.savefig('secondary_eclipse_plot.svg', transparent = True)



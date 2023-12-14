import numpy as np
import matplotlib.pyplot as plt

import exoring_objects
import scattering
import materials
import socket

AU = 1.495978707e11
L_SUN = 3.828e26
R_JUP = 6.9911e7
R_SUN = 6.957e8
JUP_TO_AU = AU / R_JUP
SUN_TO_JUP = R_SUN / R_JUP
AU_TO_SUN = AU / R_SUN
AU_TO_JUP = AU / R_JUP

bandpass = (2e-5, 5e-5)
star = exoring_objects.Star(L_SUN, R_SUN, 0.1*AU, 1.)

material = materials.RingMaterial('materials/saturn_rings.inp', 361, 500)
sc_ring = scattering.WavelengthDependentScattering(material, bandpass, star.planck_function)
#sc_ring = scattering.Rayleigh(0.1)
sc_planet = scattering.Lambert(1.)

ringed_planet = exoring_objects.RingedPlanet(sc_planet, R_JUP, sc_ring, 2*R_JUP, 5*R_JUP, [1., 1., 0.1], star)

alphas = np.linspace(-np.pi, np.pi, 10000)

light_curve = ringed_planet.light_curve(alphas)/star.luminosity#/star.L_wav(bandpass)
ring_light_curve = ringed_planet.ring.light_curve(alphas)/star.luminosity
planet_light_curve = light_curve - ring_light_curve

<<<<<<< Updated upstream
if socket.gethostname() == 'LAPTOP-2NDLGNMT':
    plt.style.use('the_usual.mplstyle')
else:
    plt.style.use('the_usual')
plt.plot(alphas, light_curve)
=======
plt.style.use('barbie')
plt.title('This Barbie is a ringed planet')
plt.plot(alphas, light_curve)
plt.plot(alphas, planet_light_curve)
plt.plot(alphas, ring_light_curve)
>>>>>>> Stashed changes

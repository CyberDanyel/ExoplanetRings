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

bandpass = (1.0e-6, 1.1e-6)
star = exoring_objects.Star(L_SUN, SUN_TO_JUP, 0.1*AU_TO_JUP, 1.)

material = materials.RingMaterial('dustkapscatmat.inp', 361, 500)
sc_ring = scattering.WavelengthDependentScattering(material, bandpass, star.planck_function)

sc_ring_diffuse = scattering.Lambert(sc_ring.albedo)
sc_planet = scattering.Rayleigh(1.)

empirical_rplanet = exoring_objects.RingedPlanet(sc_planet, 1, sc_ring, 1.2, 2., [1., .5, 0.2], star)
diffuse_rplanet = exoring_objects.RingedPlanet(sc_planet, 1, sc_ring_diffuse, 1.2, 2., [1., .5, .2], star)


alphas = list(np.linspace(-np.pi, -0.1, 1000)) + list(np.linspace(-.1, .1, 1000)) + list(np.linspace(0.1, np.pi, 1000))
alphas = np.array(alphas)


light_curve = empirical_rplanet.light_curve(alphas)/star.luminosity#/star.L_wav(bandpass)
light_curve_diffuse = diffuse_rplanet.light_curve(alphas)/star.luminosity
light_curve_planet = light_curve_diffuse - diffuse_rplanet.ring.light_curve(alphas)/star.luminosity

if socket.gethostname() == 'LAPTOP-2NDLGNMT':
    plt.style.use('the_usual.mplstyle')
else:
    plt.style.use('the_usual')
plt.plot(alphas, light_curve)
plt.plot(alphas, light_curve_diffuse)
plt.plot(alphas, light_curve_planet)

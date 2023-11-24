import numpy as np
import matplotlib.pyplot as plt

import exoring_objects
import scattering
import materials

AU = 1.495978707e11
L_SUN = 3.828e26
R_JUP = 6.9911e6
R_SUN = 6.957e8
JUP_TO_AU = AU / R_JUP
SUN_TO_JUP = R_SUN / R_JUP
AU_TO_SUN = AU / R_SUN
AU_TO_JUP = AU / R_JUP

bandpass = (1.0e-6, 1.25e-6)
star = exoring_objects.Star(L_SUN, R_SUN, 0.1*AU, 1.)

material = materials.RingMaterial('dustkapscatmat.inp', 1081, 500)
sc_ring = scattering.WavelengthDependentScattering(material, bandpass, star.planck_function)
sc_planet = scattering.Jupiter(.1)


ringed_planet = exoring_objects.RingedPlanet(sc_planet, R_JUP, sc_ring, 1.5*R_JUP, 3.*R_JUP, [1., 1., 0.1], star)

alphas = list(np.linspace(-np.pi, -.1, 2000)) + list(np.linspace(-.1, .1, 5000)) + list(np.linspace(.1, np.pi, 2000))

light_curve = ringed_planet.ring.light_curve(alphas)/star.L_wav(bandpass)

plt.style.use('the_usual')
plt.plot(alphas, light_curve)


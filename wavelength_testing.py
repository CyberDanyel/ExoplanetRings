import numpy as np
import matplotlib.pyplot as plt

import exoring_objects
import scattering
import materials

AU = 1.495978707e11
L_SUN = 3.828e26
R_JUP = 6.9911e7
R_SUN = 6.957e8
JUP_TO_AU = AU / R_JUP
SUN_TO_JUP = R_SUN / R_JUP
AU_TO_SUN = AU / R_SUN
AU_TO_JUP = AU / R_JUP

bandpass = (1.0e-6, 1.25e-6)
star = exoring_objects.Star(L_SUN, SUN_TO_JUP, 0.1*AU_TO_JUP, 1.)

material = materials.RingMaterial('materials/saturn_ring.inp', 361, 500)
#sc_ring = scattering.WavelengthDependentScattering(material, bandpass, star.planck_function)
sc_ring = scattering.Rayleigh(1.)
sc_planet = scattering.Lambert(1.)

ringed_planet = exoring_objects.RingedPlanet(sc_planet, 4, sc_ring, 5, 20, [1., -.1, 0.], star)

alphas = np.linspace(-np.pi, np.pi, 100000)

light_curve = ringed_planet.ring.light_curve(alphas)/star.luminosity#/star.L_wav(bandpass)

plt.style.use('the_usual')
plt.plot(alphas, light_curve)


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
M_JUP = 1.898e27
bandpass = (2.0e-6, 3.0e-6)
star = exoring_objects.Star(L_SUN, R_SUN, 0.1*AU, 1.)

ice = materials.RingMaterial('materials/saturn_small_ice.inp', 361, 500)
silicate = materials.RingMaterial('materials/silicate_small.inp', 361, 500)
atmos = materials.Atmosphere([500, 0.5, 0.6, 1e6], [scattering.Rayleigh, M_JUP, R_JUP], star)

sc_ice = scattering.WavelengthDependentScattering(ice, bandpass, star.planck_function)
sc_sil = scattering.WavelengthDependentScattering(silicate, bandpass, star.planck_function)
sc_ring_diffuse = scattering.Lambert(sc_sil.albedo)
sc_planet = scattering.WavelengthDependentScattering(atmos, bandpass, star.planck_function)
sc_mie = scattering.Mie(1., 10, 1.5+.1j)

ring_params = [1.2*R_JUP, 2*R_JUP, [1., 0.5, 0.1], star]

sil_rplanet = exoring_objects.RingedPlanet(sc_planet, R_JUP, sc_sil, *ring_params)
ice_rplanet = exoring_objects.RingedPlanet(sc_planet, R_JUP, sc_ice, *ring_params)
diffuse_rplanet = exoring_objects.RingedPlanet(sc_planet, R_JUP, sc_ring_diffuse, *ring_params)
mie_rplanet = exoring_objects.RingedPlanet(sc_planet, R_JUP, sc_mie, *ring_params)


alphas = list(np.linspace(-np.pi, -0.1, 1000)) + list(np.linspace(-.1, .1, 1000)) + list(np.linspace(0.1, np.pi, 1000))
alphas = np.array(alphas)


light_curve_ice = ice_rplanet.light_curve(alphas)/star.luminosity#/star.L_wav(bandpass)
light_curve_sil = sil_rplanet.light_curve(alphas)/star.luminosity
light_curve_diffuse = diffuse_rplanet.light_curve(alphas)/star.luminosity
light_curve_mie = mie_rplanet.light_curve(alphas)/star.luminosity
light_curve_planet = light_curve_diffuse - diffuse_rplanet.ring.light_curve(alphas)/star.luminosity

#if socket.gethostname() == 'LAPTOP-2NDLGNMT':
#    plt.style.use('the_usual.mplstyle')
#else:

#plt.plot(alphas, light_curve_ice)
fig, ax = plt.subplots(1, 1, figsize = (6, 4))
plt.style.use('the_usual')
plt.plot(alphas, light_curve_planet, label = 'ringless planet')
plt.plot(alphas, light_curve_sil, label = 'silicate ring,\n optool')
#plt.plot(alphas, light_curve_diffuse)
plt.plot(alphas, light_curve_mie, label = 'silicate ring,\n Mie scattering')
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])

ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

plt.xlabel('Phase angle')
plt.ylabel(r'Intensity/$L_\odot$')
plt.title(r'Wavelength band 2$\mu$m-3$\mu$m')
plt.tight_layout()
plt.legend()



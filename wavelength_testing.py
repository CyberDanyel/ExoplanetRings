import numpy as np
import matplotlib.pyplot as plt

import json
import exoring_objects
import scattering
import materials
import socket

with open('constants.json') as json_file:
    constants = json.load(json_file)

plt.style.use('poster')
bandpass = (10e-6, 14e-6)
star = exoring_objects.Star(4800, constants['R_SUN_TO_R_JUP'], .1*constants['AU_TO_R_JUP'], 1.)

ice = materials.RingMaterial('materials/saturn_small_ice.inp', 361, 500)
silicate = materials.RingMaterial('materials/silicate_small.inp', 361, 500)
atmos = materials.Atmosphere(scattering.Jupiter,  [constants['M_JUP'], 1], star, meters_per_length_unit = constants['R_JUP'])

sc_ice = scattering.WavelengthDependentScattering(ice, bandpass, star.planck_function)
sc_sil = scattering.WavelengthDependentScattering(silicate, bandpass, star.planck_function)
sc_ring_diffuse = scattering.Lambert(sc_sil.albedo)
sc_planet = scattering.WavelengthDependentScattering(atmos, bandpass, star.planck_function)
sc_mie = scattering.Mie(1., 100e-6/np.mean(bandpass), 1.5+.1j)

ring_params = [1.2, 2, [1., 0.5, 0.1], star]

sil_rplanet = exoring_objects.RingedPlanet(sc_planet, 1, sc_sil, *ring_params)
ice_rplanet = exoring_objects.RingedPlanet(sc_planet, 1, sc_ice, *ring_params)
diffuse_rplanet = exoring_objects.RingedPlanet(sc_planet, 1, sc_ring_diffuse, *ring_params)
mie_rplanet = exoring_objects.RingedPlanet(sc_planet, 1, sc_mie, *ring_params)


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

plt.plot(alphas, light_curve_planet, label = 'ringless planet', zorder=200)
plt.plot(alphas, light_curve_sil, label = 'silicate ring,\n optool')
#plt.plot(alphas, light_curve_diffuse)
plt.plot(alphas, light_curve_mie, label = 'silicate ring,\n Mie scattering')
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])

ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

plt.xlabel('Phase angle')
plt.ylabel(r'Intensity/$L_\odot$')
plt.title(r'Wavelength band %.1f$\mathrm{\mu}$m-%.1f$\mathrm{\mu}$m'%tuple(val*1e6 for val in bandpass))
plt.tight_layout()
plt.legend()



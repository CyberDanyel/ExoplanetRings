# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:18:17 2024

@author: victo
"""

import numpy as np
import matplotlib.pyplot as plt
import scattering
import materials
import exoring_objects

AU = 1.495978707e11
L_SUN = 3.828e26
R_JUP = 6.9911e7
R_SUN = 6.957e8
JUP_TO_AU = AU / R_JUP
SUN_TO_JUP = R_SUN / R_JUP
AU_TO_SUN = AU / R_SUN
AU_TO_JUP = AU / R_JUP
M_JUP = 1.898e27

data = np.loadtxt('C:/users/victo/onedrive/desktop/jup_data.txt')
jup_wavs = data[:,0]
jup_albedo = data[:,3]
nep_albedo = data[:,6]
titan_albedo = data[:,7]

star = exoring_objects.Star(5780, R_SUN, 1*AU, 1.)
atmos = materials.Atmosphere(scattering.Jupiter,  [M_JUP, R_JUP], star, invert=False)
silicate = materials.RingMaterial('materials/silicate_small.inp', 361, 500)



plt.style.use('poster')

#albedo ratio plot
fig1, ax1 = plt.subplots(1, 1)
min_wav = max(min(atmos.wavelengths), min(silicate.wavelengths))
max_wav = min(max(atmos.wavelengths), max(silicate.wavelengths))
wavs = np.linspace(min_wav, max_wav, 1000)
ax1.plot(wavs*1e6, silicate.albedo(wavs)/atmos.albedo(wavs))
ax1.set_xlabel(r'Wavelength ($\mu$m)')
ax1.set_ylabel(r'Ring albedo/atmospheric albedo')
plt.tight_layout()




# comparison to actual planets
# using closest possible values for Jupiter
fig2, ax2 = plt.subplots(1, 1, figsize = (5, 3))
jup_wavs*=1e-3

ax2.plot(jup_wavs, atmos.albedo(jup_wavs*1e-6), label = r'Platon', zorder = 5)
ax2.plot(jup_wavs, jup_albedo, label = r'Jupiter')
ax2.plot(jup_wavs, nep_albedo, label = r'Neptune')
ax2.plot(jup_wavs, titan_albedo, label = r'Titan')
ax2.set_xlabel(r'Wavelength ($\mathrm{\mu}$m)')
ax2.set_ylabel(r'Albedo')
ax2.legend(loc = 'lower left')
plt.tight_layout()
plt.savefig('c:/users/victo/onedrive/desktop/msci project/viva/solar_system_spectra.svg', transparent = True)

#just silicate
fig3, ax3 = plt.subplots(1, 1)

ax3.plot(silicate.wavelengths*1e6, silicate.albedo(silicate.wavelengths))
ax3.set_xlabel(r'Wavelength ($\mu$m)')
ax3.set_ylabel(r'Ring albedo')
plt.tight_layout()



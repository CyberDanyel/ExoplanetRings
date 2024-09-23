import exoring_objects
import materials
import scattering
import json
import numpy as np
import pandas as pd
import scipy.optimize as op
import matplotlib.pyplot as plt
from tqdm import tqdm

with open('constants.json') as json_file:
    constants = json.load(json_file)

alphas = np.linspace(-np.pi, np.pi, 10000)  # adjust sampling across phase here
real_planet_info = pd.read_csv('exoplanet_info.csv', header=29)


def get_asymmetry(exoring_object, alpha):
    """
    Finds the asymmetry for a Planet, RingedPlanet or Ring object as a function of phase angle
    """
    front_half = exoring_object.light_curve(-np.abs(alpha))/exoring_object.star.luminosity
    back_half = exoring_object.light_curve(np.abs(alpha))/exoring_object.star.luminosity
    asymmetry = back_half - front_half
    return asymmetry


def get_max_asymmetry(exoring_object):
    """
    Find the maximum asymmetry for a given object
    """
    min_result = op.minimize(lambda x: -get_asymmetry(exoring_object, x), x0=np.array([np.pi/2]))
    alpha = min_result.x
    max_asymmetry = exoring_object.light_curve(alpha)/exoring_object.star.luminosity
    return float(max_asymmetry)


def run_test_planet_geometry():
    """
    Generate the light curve of a test planet using the geometric model
    """
    # instantiating the star - using the Sun as an example
    sun = exoring_objects.Star(constants['T_SUN'], constants['R_SUN'], 0.2*constants['AU'], constants['M_SUN'])
    planet_sc = scattering.Lambert(albedo=1.)  # instantiating a scattering law to pass to the planet later
    ring_sc = scattering.Lambert(albedo=0.8)  # ditto for the ring

    planet = exoring_objects.RingedPlanet(planet_sc, constants['R_JUP'],
                                          ring_sc, constants['R_JUP'], 4*constants['R_JUP'],
                                          [np.sqrt(2), np.sqrt(2), 0.], sun)  # instantiating the planet

    return planet.light_curve(alphas)/sun.luminosity  # returns a 1D array with the light curve value for each alpha


def run_test_planet_full():
    """
    Generate the light curve of a test planet using the full model
    """
    bandpass = (11.43e-6, 14.17e-6)  # bandpass of the f1280w filter on JWST
    # instantiating the star - using the Sun as an example
    sun = exoring_objects.Star(constants['T_SUN'], constants['R_SUN'], 0.1*constants['AU'], constants['M_SUN'])

    # instantiating the material objects that are later passed to the scattering code
    # this retrieves the properties of materials from outside sources before using them,
    # atmosphere properties from Platon (Zhang et al. 2019)
    # and ring particle properties from optool (Dominik et al. 2021)
    planet_atmos = materials.Atmosphere(scattering.Jupiter, [constants['M_JUP'], constants['R_JUP']], sun)
    ring_material = materials.RingMaterial('materials/silicate_small.inp', nang=361, nlam=500)

    # instantiating scattering law objects to pass to the geometric model
    planet_sc = scattering.WavelengthDependentScattering(planet_atmos, bandpass, sun.planck_function)
    ring_sc = scattering.WavelengthDependentScattering(ring_material, bandpass, sun.planck_function)

    # finally instantiating the planet
    planet = exoring_objects.RingedPlanet(planet_sc, constants['R_JUP'],
                                          ring_sc, constants['R_JUP'], 2*constants['R_JUP'],
                                          [1., 1., 0.], sun)

    return planet.light_curve(alphas)/sun.luminosity  # generates the light curve and normalizes to stellar luminosity


light_curve = run_test_planet_full()
plt.plot(alphas, light_curve)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
plt.xlabel(r'Phase angle $\alpha$')
plt.ylabel(r'Reflected light$/L_\star$')
plt.show()

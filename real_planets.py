# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:56:23 2024

@author: victo
"""

# REAL planets

import numpy as np
import matplotlib.pyplot as plt

import json
import exoring_objects
import scattering
import materials
import pandas as pd
#import fitting

with open('constants.json') as json_file:
    constants = json.load(json_file)

plt.style.use('the_usual')
bandpass = (11.43e-6, 14.17e-6) # f1280w
R_SUN = constants['R_SUN']
M_SUN = constants['M_SUN']
AU = constants['AU']
R_JUP = constants['R_JUP']
R_EARTH = constants['R_EARTH']
M_EARTH = constants['M_EARTH']
RHO_JUP = constants['RHO_JUP']
alphas = np.linspace(-np.pi, np.pi, 10000)
phis = [np.pi/16, np.pi/8, 3*np.pi/16, np.pi/4, 5*np.pi/16]
phi_labels = [r'$\frac{\pi}{16}$', r'$\frac{\pi}{8}$', r'$\frac{3\pi}{16}$', r'$\frac{\pi}{4}$', r'$\frac{5\pi}{16}$']

silicate = materials.RingMaterial('materials/silicate_small.inp', 361, 500)

def run_real_planet(name):
    planets = pd.read_csv('exoplanet_info.csv', header = 29)
    planet_index = planets.index[planets['pl_name'] == name][0]
    planet = planets.iloc[planet_index]
    T_star = planet['st_teff']
    R_star = planet['st_rad']*R_SUN
    distance = planet['pl_orbsmax']*AU
    M_star = planet['st_mass']*M_SUN
    M = planet['pl_bmasse']*M_EARTH
    R = planet['pl_radj']*R_JUP
    density = 3*M/(4*np.pi*R**3)
    R_reduced = R*(density/RHO_JUP)**(1/3)
    saturn_J2 = .016298


    R_hill = distance * (M / (3*M_star))**(1/3)
    R_roche = R_reduced * 2.45 * (RHO_JUP/2090)**(1/3) # using the lower limit of silicate used by Piro and Vissapragada
    R_laplace = (2*saturn_J2*R_reduced**2*distance**3*(M/M_star))**(1/5)
    R_max = np.array([R_roche, R_laplace])
    max_phis = np.arccos(((R**2-R_reduced**2)/(R_max**2-R_reduced**2)))

    # defining objects
    star = exoring_objects.Star(T_star, R_star, distance, M_star)
    atmos = materials.Atmosphere(scattering.Jupiter, [M, R], star)
    atmos_sc = scattering.WavelengthDependentScattering(atmos, bandpass, star.planck_function)
    ring_sc = scattering.WavelengthDependentScattering(silicate, bandpass, star.planck_function)

    # calculate light curves
    unringed_planet = exoring_objects.Planet(atmos_sc, R, star)
    unringed_light_curve = unringed_planet.light_curve(alphas)
    ringed_light_curves = np.zeros((len(phis), len(alphas)))
    max_light_curves = np.zeros((len(max_phis), len(alphas)))
    for n, phi in enumerate(phis):
        normal = [np.cos(phi), np.sin(phi), 0]
        ringed_planet = exoring_objects.RingedPlanet(atmos_sc, R_reduced, ring_sc, R_reduced, R/np.sqrt(np.cos(phi)), normal, star)
        ringed_light_curves[n] += ringed_planet.light_curve(alphas)
    for n, phi in enumerate(max_phis):
        normal = [np.cos(phi), np.sin(phi), 0]
        ringed_planet = exoring_objects.RingedPlanet(atmos_sc, R_reduced, ring_sc, R_reduced, R / np.sqrt(np.cos(phi)), normal, star)
        max_light_curves[n] += ringed_planet.light_curve(alphas)

    return unringed_light_curve/star.luminosity, ringed_light_curves/star.luminosity, max_light_curves/star.luminosity

def light_curve_plots(planetname):
    unringed_light_curve, ringed_light_curves, max_light_curves = run_real_planet(planetname)
    fig, ax = plt.subplots(1, 1, figsize=[5, 2.5])
    linestyles = ['-.', '-']
    labels = [r'$R_{max}=$Roche limit', r'$R_{max}=$Laplace radius']
    for n, phi_curve in enumerate(ringed_light_curves):
        plt.plot(alphas, phi_curve, label=phi_labels[n], linestyle=':', color=[n / len(ringed_light_curves), 0, 1 - n / len(ringed_light_curves)])
    for n, max_curve in enumerate(max_light_curves):
        plt.plot(alphas, max_curve, label=labels[n], linestyle=linestyles[n],
                 color='xkcd:midnight')
    plt.plot(alphas, unringed_light_curve, label='Inflated planet', color='xkcd:carolina blue')
    plt.ylabel(r'$L(\alpha)/L_{\odot}$')
    plt.xlabel(r'Phase angle $\alpha$')
    plt.xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    fig.legend(handlelength=.8)
    plt.tight_layout()
    plt.savefig(f'{planetname}_light_curves.svg', transparent=True)
    #plt.show()

def max_residual_plots(planetname):
    unringed_light_curve, ringed_light_curves, max_light_curves = run_real_planet(planetname)
    max_residuals = max_light_curves - np.broadcast_to(unringed_light_curve, np.shape(max_light_curves))
    phi_residuals = ringed_light_curves - np.broadcast_to(unringed_light_curve, np.shape(ringed_light_curves))
    fig, ax = plt.subplots(1, 1, figsize=[5, 2.5])
    labels = [r'$R_{max}=$Roche limit', r'$R_{max}=$Laplace radius']
    linestyles = ['-.', '-']
    for n, phi_residual in enumerate(phi_residuals):
        plt.plot(alphas, phi_residual, label=phi_labels[n], linestyle=':',
                 color=[n / len(phi_residuals), 0, 1 - n / len(phi_residuals)])
    for n, max_residual in enumerate(max_residuals):
        plt.plot(alphas, max_residual, label=labels[n], linestyle=linestyles[n],
                 color='xkcd:midnight')
    plt.ylabel(r'Residuals/$L_{\odot}$')
    plt.xlabel(r'Phase angle $\alpha$')
    plt.xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.legend(handlelength=.8)
    plt.tight_layout()
    plt.savefig(f'{planetname}_residuals.svg', transparent=True)
    #plt.show()
def asymm_from_light_curves(light_curves):
    front_half = light_curves[:, :int(len(alphas) / 2)]
    back_half = light_curves[:, int(len(alphas) / 2):]
    asymmetries = back_half - front_half[:, ::-1]  # only works for an even number of alphas
    return asymmetries
def max_asymm_plots(planetname):
    unringed_light_curve, ringed_light_curves, max_light_curves = run_real_planet(planetname)
    max_residuals = asymm_from_light_curves(max_light_curves)
    phi_residuals = asymm_from_light_curves(ringed_light_curves)
    fig, ax = plt.subplots(1, 1, figsize=[5, 2.5])
    labels = [r'$R_{max}=$Roche limit', r'$R_{max}=$Laplace radius']
    linestyles = ['-.', '-']
    for n, phi_residual in enumerate(phi_residuals):
        plt.plot(alphas[int(len(alphas) / 2):], phi_residual, label=phi_labels[n], linestyle=':', color=[n/len(phi_residuals), 0, 1-n/len(phi_residuals)])
    for n, max_residual in enumerate(max_residuals):
        plt.plot(alphas[int(len(alphas) / 2):], max_residual, label=labels[n], linestyle=linestyles[n], color='xkcd:midnight')
    plt.ylabel(r'Asymmetry$/L_{\star}$')
    plt.xlabel(r'Phase angle $\alpha$')
    plt.xlim(0, np.pi)
    ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    plt.legend(handlelength=.8, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{planetname}_max_asymmetry.svg', transparent = True)
    #plt.show()


def all_laplace_asymmetries():
    planets = pd.read_csv('exoplanet_info.csv', header=29)
    fig, ax = plt.subplots(1, 1, figsize=[5, 3])
    for n, planet in enumerate(planets.iloc):
        if n == 16:
            break
        T_star = planet['st_teff']
        R_star = planet['st_rad'] * R_SUN
        distance = planet['pl_orbsmax'] * AU
        M_star = planet['st_mass'] * M_SUN
        M = planet['pl_bmasse'] * M_EARTH
        R = planet['pl_radj'] * R_JUP
        density = planet['pl_dens'] * 1000  # gcm^-3 to kgm^-3
        R_reduced = R * (density / RHO_JUP) ** (1 / 3)
        saturn_J2 = .016298

        R_hill = distance * (M / (3 * M_star)) ** (1 / 3)
        R_roche = R_reduced * 2.45 * (RHO_JUP/2090) ** (1 / 3)  # the density of silicate is about 3 times that of Jupiter
        R_laplace = (2 * saturn_J2 * R_reduced ** 2 * distance ** 3 * (M / M_star)) ** (1 / 5)
        R_max = R_laplace
        phi_max = np.arccos((R**2 - R_reduced**2) / (R_max**2 - R_reduced**2))

        # defining objects
        star = exoring_objects.Star(T_star, R_star, distance, M_star)
        try:
            atmos = materials.Atmosphere(scattering.Jupiter, [M, R_reduced], star)
        except AssertionError:
            raise AssertionError(planet['pl_name'])
        atmos_sc = scattering.WavelengthDependentScattering(atmos, bandpass, star.planck_function)
        ring_sc = scattering.WavelengthDependentScattering(silicate, bandpass, star.planck_function)

        # calculate light curves
        unringed_planet = exoring_objects.Planet(atmos_sc, R, star)
        unringed_light_curve = unringed_planet.light_curve(alphas)/star.luminosity
        normal = [np.cos(phi_max), np.sin(phi_max), 0]
        ringed_planet = exoring_objects.RingedPlanet(atmos_sc, R_reduced, ring_sc, R_reduced, R / np.sqrt(np.cos(phi_max)), normal, star)
        max_light_curve = ringed_planet.light_curve(alphas)
        max_light_curve /= star.luminosity
        # calculate asymmetries
        front_half = max_light_curve[:int(len(alphas) / 2)]
        back_half = max_light_curve[int(len(alphas) / 2):]
        residuals = back_half - front_half[::-1]  # only works for an even number of alphas
        if planet['pl_name'] == 'Kepler-435 b':
            ax.plot(alphas[int(len(alphas) / 2):], residuals.flatten(), label=planet['pl_name'], linewidth = 2, color = 'xkcd:prussian blue',zorder = 10000)
        else:
            max_residue = max(residuals.flatten())
            ax.plot(alphas[int(len(alphas) / 2):], residuals.flatten(), color=[1-0.5*(max_residue/2e-6) for i in range(3)], zorder = int(max_residue/1e-7))
        plt.ylabel(r'Asymmetry$/L_{\star}$')
        plt.xlabel(r'Phase angle $\alpha$')
        plt.xlim(0, np.pi)
        ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        ax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
        plt.legend(loc = 'center')
        plt.tight_layout()
    plt.savefig('all_laplace_asymmetries.svg', transparent = True)

def generate_tables():
    # finds the maximum possible asymmetry for each planet with an upper bound on radius being the Laplace radius
    planets = pd.read_csv('exoplanet_info.csv', header=29)
    input_array = np.zeros((16, 6)).astype(str)
    output_array = np.zeros((16, 7)).astype(str)
    star_array = np.zeros((16, 5)).astype(str)
    for n, planet in enumerate(planets.iloc):
        if n == 16:
            break
        T_star = planet['st_teff']
        R_star = planet['st_rad'] * R_SUN
        distance = planet['pl_orbsmax'] * AU
        M_star = planet['st_mass'] * M_SUN
        M = planet['pl_bmasse'] * M_EARTH
        R = planet['pl_radj'] * R_JUP
        density = planet['pl_dens'] * 1000  # gcm^-3 to kgm^-3
        R_reduced = R * (density / RHO_JUP) ** (1 / 3)
        saturn_J2 = .016298

        R_hill = distance * (M / (3 * M_star)) ** (1 / 3)
        R_roche = R_reduced * 2.45 * (RHO_JUP / 2090) ** (
                    1 / 3)  # the density of silicate is about 3 times that of Jupiter
        R_laplace = (2 * saturn_J2 * R_reduced ** 2 * distance ** 3 * (M / M_star)) ** (1 / 5)
        R_max = np.array([R_roche, R_laplace])
        phi_max = np.arccos((R ** 2 - R_reduced ** 2) / (R_max ** 2 - R_reduced ** 2))

        # defining objects
        star = exoring_objects.Star(T_star, R_star, distance, M_star)
        try:
            atmos = materials.Atmosphere(scattering.Jupiter, [M, R_reduced], star)
        except AssertionError:
            raise AssertionError(planet['pl_name'])
        atmos_sc = scattering.WavelengthDependentScattering(atmos, bandpass, star.planck_function)
        ring_sc = scattering.WavelengthDependentScattering(silicate, bandpass, star.planck_function)

        # calculate light curves
        unringed_planet = exoring_objects.Planet(atmos_sc, R, star)
        unringed_light_curve = unringed_planet.light_curve(alphas) / star.luminosity

        for i, phi in enumerate(phi_max):
            normal = [np.cos(phi), np.sin(phi), 0]
            ringed_planet = exoring_objects.RingedPlanet(atmos_sc, R_reduced, ring_sc, R_reduced,
                                                     R / np.sqrt(np.cos(phi)), normal, star)
            max_light_curve = ringed_planet.light_curve(alphas)
            max_light_curve /= star.luminosity
            # calculate asymmetries
            front_half = max_light_curve[:int(len(alphas) / 2)]
            back_half = max_light_curve[int(len(alphas) / 2):]
            residuals = back_half - front_half[::-1]  # only works for an even number of alphas
            if i == 0:
                roche_max_asymm = max(residuals)
            elif i == 1:
                laplace_max_asymm = max(residuals)

        #placing results in tables
        input_array[n][0] = planet['pl_name'] # Name
        input_array[n][1] = np.round(M/M_EARTH, 2) # mass
        input_array[n][2] = np.round(R/R_JUP, 2) # radius
        input_array[n][3] = np.round(density/1000, 2)# density
        input_array[n][4] = np.round(distance/AU, 2) # semi-major axis
        input_array[n][5] = planet['pl_refname'] # data reference

        star_array[n][0] = planet['hostname'] # Name
        star_array[n][1] = T_star # effective temperature
        star_array[n][2] = np.round(R_star/R_SUN, 2) # stellar radius
        star_array[n][3] = np.round(M_star/M_SUN, 2) # stellar mass
        star_array[n][4] = planet['st_refname']

        output_array[n][0] = planet['pl_name'] # Name
        output_array[n][1] = np.round(R/R_JUP, 2) # inferred radius
        output_array[n][2] = np.round(R_reduced/R, 2) # reduced radius
        output_array[n][3] = np.round(R_roche/R, 2) # Roche limit
        output_array[n][4] = np.round(R_laplace/R, 2) # Laplace radius
        if not np.isnan(roche_max_asymm):
            output_array[n][5] = np.round(roche_max_asymm / 1e-7, 2)
        else:
            output_array[n][5] = '(n/a)'

        if not np.isnan(laplace_max_asymm):
            output_array[n][6] = np.round(laplace_max_asymm / 1e-7, 2)
        else:
            output_array[n][6] = '(n/a)'

    input_columns = ['Name', 'Mass', 'Radius', 'Density', 'Semi-major axis', 'Reference']
    input_table = pd.DataFrame(data=input_array, columns=input_columns)
    output_columns = ['Name', 'Inferred Radius', 'Reduced Radius', 'Roche limit', 'Laplace radius', 'Roche Asymm', 'Laplace Asymm']
    output_table = pd.DataFrame(data=output_array, columns=output_columns)
    star_columns = ['Name', 'Temp', 'Mass', 'Radius', 'Reference']
    star_table = pd.DataFrame(data=star_array, columns=star_columns)

    input_table_string = input_table.to_latex()
    output_table_string = output_table.to_latex()
    star_table_string = star_table.to_latex()
    with open('input_table.txt', 'w') as input_file:
        input_file.write(input_table_string)
    with open('output_table.txt', 'w') as output_file:
        output_file.write(output_table_string)
    with open('star_table.txt', 'w') as star_file:
        star_file.write(star_table_string)


#max_residual_plots('TOI-5398 b')
#max_asymm_plots('TOI-5398 b')
#light_curve_plots('TOI-5398 b')
#all_laplace_asymmetries()
#max_asymm_plots('Kepler-18 d')
#max_asymm_plots('Kepler-435 b')
generate_tables()
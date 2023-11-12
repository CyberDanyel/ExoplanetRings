# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:37:10 2023

@author: victo
"""

# fitting

import numpy as np
import matplotlib.pyplot as plt
import exoring_objects
import scattering
import scipy.optimize as op

AU = 1.495978707e13
L_SUN = 3.828e33
R_JUP = 6.9911e9
R_SUN = 6.957e10
AU_TO_JUP = AU / R_JUP
SUN_TO_JUP = R_SUN / R_JUP
SUN_TO_AU = AU / R_SUN


class FittingPlanet(exoring_objects.Planet):
    def __init__(self, sc_law, star, *params):
        radius = params[0]
        sc_args = params[6:6 + len(sc_law.__init__.__code__.co_varnames) - 1] # This doesn't work for just fitting the planet,as we only give radius as a basic variable, FIX
        try:
            sc = sc_law(*sc_args)
        except TypeError:
            raise TypeError('Too little/many arguments for planet sc law')

        exoring_objects.Planet.__init__(self, sc, radius, star)


class FittingRingedPlanet(exoring_objects.RingedPlanet, FittingPlanet):
    def __init__(self, planet_sc_law, ring_sc_law, star, *params):
        FittingPlanet.__init__(self, planet_sc_law, star, *params)
        inner_rad, ring_width, n_x, n_y, n_z = params[1:6]
        ring_sc_args = params[6 + len(planet_sc_law.__init__.__code__.co_varnames) - 1:]
        try:
            ring_sc = ring_sc_law(*ring_sc_args)
        except TypeError:
            raise TypeError('Too little/many arguments for ring sc law')

        exoring_objects.RingedPlanet.__init__(self, self.sc_law, self.radius, ring_sc, inner_rad,
                                              inner_rad + ring_width, [n_x, n_y, n_z], star)


def gaussian(x, mu, sigma):
    with np.errstate(under='ignore'):
        return (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def log_likelihood_ring(data, planet_sc_law, ring_sc_law, star, *params):
    alpha = data[0]
    I = data[1]
    I_errs = data[2]
    model_ringed_planet = FittingRingedPlanet(planet_sc_law, ring_sc_law, star, *params)
    x = model_ringed_planet.light_curve(alpha)
    with np.errstate(divide='raise'):
        try:
            print(-np.sum(np.log(gaussian(x, I, I_errs))))
            return -np.sum(np.log(gaussian(x, I, I_errs)))
        except:  # The gaussian has returned 0 for at least 1 data point
            with np.errstate(divide='ignore'):
                print('Triggered')
                logs = np.log(gaussian(x, I, I_errs))
                for index, element in enumerate(logs):
                    if np.isinf(element):
                        logs[index] = -1000000000000
                print(-np.sum(logs))
                return -np.sum(logs)


def log_likelihood_planet(data, sc_law, star, *params):
    alpha = data[0]
    I = data[1]
    I_errs = data[2]
    model_planet = FittingPlanet(sc_law, star, *params)
    x = model_planet.light_curve(alpha)
    return -np.sum(np.log(gaussian(x, I, I_errs)))


def minimize_logL_ring(data, planet_sc_law, ring_sc_law, star, p0, bounds):
    m = op.minimize(lambda p: log_likelihood_ring(data, planet_sc_law, ring_sc_law, star, *p), p0, bounds=bounds)
    return m.x


def minimize_logL_planet(data, sc_law, star, p0, bounds):
    m = op.minimize(lambda p: log_likelihood_planet(data, sc_law, star, *p), p0, bounds=bounds)
    return m.x


def fit_data_planet(data, planet_sc_law, star, init_guess: dict):
    try:
        radius = init_guess['radius'][0]
        radius_bounds = init_guess['radius'][1]
        planet_sc_args = init_guess['planet_sc_args'][0]
        planet_sc_args_bounds = init_guess['planet_sc_args'][1]
    except KeyError:
        raise KeyError('Not all required parameters were inputted')

    p0 = [radius, *planet_sc_args]
    bounds = [radius_bounds, *planet_sc_args_bounds]

    result = minimize_logL_planet(data, planet_sc_law, star, p0, bounds)
    output = dict()
    output['radius'], output['inner_rad'], output['ring_width'], n_x, n_y, n_z, output['planet_sc_args'], output[
        'ring_sc_args'] = result
    output['ring_normal'] = (n_x, n_y, n_z)
    return output


def fit_data_ring(data, planet_sc_law, ring_sc_law, star, init_guess: dict):
    try:
        radius = init_guess['radius'][0]
        radius_bounds = init_guess['radius'][1]
        inner_rad = init_guess['inner_rad'][0]
        inner_rad_bounds = init_guess['inner_rad'][1]
        ring_width = init_guess['ring_width'][0]
        ring_width_bounds = init_guess['ring_width'][1]
        n_x, n_y, n_z = init_guess['ring_normal'][0]
        n_x_bounds, n_y_bounds, n_z_bounds = init_guess['ring_normal'][1]
        planet_sc_args = init_guess['planet_sc_args'][0]
        planet_sc_args_bounds = init_guess['planet_sc_args'][1]
        ring_sc_args = init_guess['ring_sc_args'][0]
        ring_sc_args_bounds = init_guess['ring_sc_args'][1]
    except KeyError:
        raise KeyError('Not all required parameters were inputted')

    p0 = [radius, inner_rad, ring_width, n_x, n_y, n_z, *planet_sc_args, *ring_sc_args]
    bounds = [radius_bounds, inner_rad_bounds, ring_width_bounds, n_x_bounds, n_y_bounds, n_z_bounds,
              *planet_sc_args_bounds, *ring_sc_args_bounds]

    result = minimize_logL_ring(data, planet_sc_law, ring_sc_law, star, p0, bounds)
    output = dict()
    output['radius'], output['inner_rad'], output['ring_width'], n_x, n_y, n_z, output['planet_sc_args'], output[
        'ring_sc_args'] = result
    output['ring_normal'] = (n_x, n_y, n_z)
    return output


def generate_data(test_planet):
    test_alphas = list(np.linspace(-np.pi, -.3, 10)) + list(np.linspace(-.25, .25, 10)) + list(
        np.linspace(.3, np.pi, 10))
    test_alphas = np.array(test_alphas)
    I = test_planet.light_curve(test_alphas)
    errs = 0.02 * I + 1e-8
    noise_vals = np.random.normal(size=len(test_alphas))
    data_vals = errs * noise_vals + I
    data = np.array([test_alphas, data_vals, errs])
    return data

#should turn fitting results into their own class later
def plot_ring_result(data, result_ringfit):
    plt.style.use('the_usual')
    fmt_list = list([result_ringfit['radius'], result_ringfit['inner_rad'], result_ringfit['ring_width'], *result_ringfit['ring_normal'], result_ringfit['planet_sc_args'], result_ringfit['ring_sc_args']])
    fitted_planet = FittingRingedPlanet(scattering.Jupiter, scattering.Rayleigh, star, *fmt_list)
    alphas = np.linspace(-np.pi, np.pi, 10000)
    plt.errorbar(data[0], data[1], data[2], fmt = '.')
    plt.plot(alphas, fitted_planet.light_curve(alphas))
    

star = exoring_objects.Star(1, SUN_TO_JUP, 0.1 * AU_TO_JUP, 1)

test_planet = exoring_objects.RingedPlanet(scattering.Jupiter(1), 1, scattering.Rayleigh(0.9), 2, 3,
                                           np.array([1., 1., 0]), star)
data = generate_data(test_planet)

init_guess_planet = {'radius': (1, (0, np.inf)), 'planet_sc_args': ([1], [(0, 1)])}
init_guess_ring = {'radius': (1, (0, np.inf)), 'inner_rad': (2, (2, np.inf)), 'ring_width': (1, (0, np.inf)),
                   'ring_normal': ([1, 1, 0], [(0, 1), (0, 1), (0, 1)]),
                   'planet_sc_args': ([1], [(0, 1)]), 'ring_sc_args': ([0.9], [(0, 1)])}

#result_planetfit = fit_data_planet(data, scattering.Jupiter, star, init_guess_ring)
result_ringfit = fit_data_ring(data, scattering.Jupiter, scattering.Rayleigh, star, init_guess_ring)
#print('planetfit', result_planetfit)
#print('ringfit', result_ringfit)
'''
bounds = [(0,np.inf),(0,np.inf),(0,np.inf),(0.01,np.inf),(0.01,np.inf),(0.01,np.inf)]
vals = fit_data_ring(data,scattering.Jupiter,scattering.Rayleigh,star,init_guess)

planet_vals = minimize_logL_planet(data, scattering.Rayleigh, star, np.array([1, 1]), bounds=[(0, np.inf), (0., np.inf)])
ring_vals = minimize_logL_ring(data, scattering.Jupiter, scattering.Rayleigh, star, np.array([.2, .5, 1., 1., 1., 1., 1., 1.]),
                               bounds=[(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0.01, np.inf),
                                       (0.01, np.inf), (0.01, np.inf)])


#alphas = np.linspace(-np.pi, np.pi, 10000)
#plt.style.use('barbie')
#plt.title('This Barbie is NOT a ringed planet')
#plt.errorbar(data[0], data[1], data[2], fmt='.')
#plt.plot(alphas, FittingPlanet('Rayleigh', star, *planet_vals).light_curve(alphas), label='Planet')
#plt.plot(alphas, FittingRingedPlanet('Jupiter', 'Rayleigh', star, *ring_vals).light_curve(alphas), label='Planet+Ring')

plt.legend()
'''

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
    def __init__(self, sc_law, star, parameters):
        radius = parameters['radius']
        sc_args = parameters['planet_sc_args']
        try:
            sc = sc_law(**sc_args)
        except TypeError:
            raise TypeError('Too little/many arguments for ring sc law')
        exoring_objects.Planet.__init__(self, sc, radius, star)


class FittingRingedPlanet(exoring_objects.RingedPlanet, FittingPlanet):
    def __init__(self, planet_sc_law, ring_sc_law, star, parameters):
        FittingPlanet.__init__(self, planet_sc_law, star, parameters)
        inner_rad, ring_width, n_x, n_y, n_z, ring_sc_args = parameters['inner_rad'], parameters['ring_width'], \
            parameters['n_x'], parameters['n_y'], parameters['n_z'], parameters['ring_sc_args']
        try:
            ring_sc = ring_sc_law(**ring_sc_args)
        except TypeError:
            raise TypeError('Too little/many arguments for ring sc law')

        exoring_objects.RingedPlanet.__init__(self, self.sc_law, self.radius, ring_sc, inner_rad,
                                              inner_rad + ring_width, [n_x, n_y, n_z], star)
        # Note that sc.law here
        # does not follow the previous naming convention, it is not the class
        # itself, but an instantiation of that class (we would usually not have
        # law in this, but it was created as such in the exoring objects code
        # from the beginning so whatever


def gaussian(x, mu, sigma):
    with np.errstate(under='ignore'):
        return (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def log_likelihood_ring(data, planet_sc_law, ring_sc_law, star, planet_sc_args_order, ring_sc_args_order, *params):
    alpha = data[0]
    I = data[1]
    I_errs = data[2]
    planet_sc_args_positional = params[6:6 + len(planet_sc_law.__init__.__code__.co_varnames) - 1]
    ring_sc_args_positional = params[6 + len(planet_sc_law.__init__.__code__.co_varnames) - 1:]
    parameters = {'radius': params[0], 'inner_rad': params[1], 'ring_width': params[2], 'n_x': params[3],
                  'n_y': params[4], 'n_z': params[5],
                  'planet_sc_args': {planet_sc_args_order[index]: planet_sc_args_positional[index] for index in
                                     range(len(planet_sc_args_positional))},
                  'ring_sc_args': {ring_sc_args_order[index]: ring_sc_args_positional[index] for index in
                                   range(len(ring_sc_args_positional))}}
    model_ringed_planet = FittingRingedPlanet(planet_sc_law, ring_sc_law, star, parameters)
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
                        logs[index] = -1000
                print(-np.sum(logs))
                return -np.sum(logs)


def log_likelihood_planet(data, sc_law, star, planet_sc_args_order, *params):
    alpha = data[0]
    I = data[1]
    I_errs = data[2]
    planet_sc_args_positional = params[1:]
    parameters = {'radius': params[0],
                  'planet_sc_args': {planet_sc_args_order[index]: planet_sc_args_positional[index] for index in
                                     range(len(planet_sc_args_positional))}}
    model_planet = FittingPlanet(sc_law, star, parameters)
    x = model_planet.light_curve(alpha)
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
                        logs[index] = -1000
                print(-np.sum(logs))
                return -np.sum(logs)


def minimize_logL_ring(data, planet_sc_law, ring_sc_law, star, p0, bounds):
    m = op.minimize(lambda p: log_likelihood_ring(data, planet_sc_law, ring_sc_law, star, *p), p0, bounds=bounds)
    return m.x


def minimize_logL_planet(data, sc_law, star, p0, bounds):
    m = op.minimize(lambda p: log_likelihood_planet(data, sc_law, star, *p), p0, bounds=bounds,method='BFGS',options={'maxiter':100000,'disp':True})
    return m.x


def fit_data_planet(data, planet_sc_law, star, init_guess: dict):
    try:
        radius = init_guess['radius'][0]
        radius_bounds = init_guess['radius'][1]
        planet_sc_args = init_guess['planet_sc_args'][0]
        planet_sc_args_bounds = init_guess['planet_sc_args'][1]
    except KeyError:
        raise KeyError('Not all required parameters were inputted')

    planet_sc_args_positional = []
    planet_sc_bounds_positional = []
    planet_sc_args_order = dict()
    for i, dictvals in enumerate(planet_sc_args.items()):
        key = dictvals[0]
        item = dictvals[1]
        if key in planet_sc_args_bounds.keys():
            planet_sc_args_positional.append(item)
            planet_sc_bounds_positional.append(planet_sc_args_bounds[key])
            planet_sc_args_order[i] = key
        else:
            raise 'Given parameter in planet_sc_args was not given a bound'
    p0 = np.array([radius, *planet_sc_args_positional])
    bounds = [radius_bounds, *planet_sc_bounds_positional]
    result = op.minimize(lambda p: log_likelihood_planet(data, planet_sc_law, star, planet_sc_args_order, *p), p0,
                         bounds=bounds).x

    output = dict()
    output['radius'] = result[0]
    planet_sc_args = result[1:]
    output['planet_sc_args'] = {planet_sc_args_order[index]: planet_sc_args[index] for index in
                                range(len(planet_sc_args))}
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

        init_guess_ring = {'radius': (1, (0, np.inf)), 'inner_rad': (2, (2, np.inf)), 'ring_width': (1, (0, np.inf)),
                           'ring_normal': ([1, 1, 0], [(0, 1), (0, 1), (0, 1)]),
                           'planet_sc_args': ({'albedo': 1}, {'albedo': (0, 1)}),
                           'ring_sc_args': ({'albedo': 1}, {'albedo': (0, 1)})}
    except KeyError:
        raise KeyError('Not all required parameters were inputted')
    planet_sc_args_positional = []
    planet_sc_bounds_positional = []
    ring_sc_args_positional = []
    ring_sc_bounds_positional = []
    planet_sc_args_order = dict()
    ring_sc_args_order = dict()
    for i, dictvals in enumerate(planet_sc_args.items()):
        key = dictvals[0]
        item = dictvals[1]
        if key in planet_sc_args_bounds.keys():
            planet_sc_args_positional.append(item)
            planet_sc_bounds_positional.append(planet_sc_args_bounds[key])
            planet_sc_args_order[i] = key
        else:
            raise 'Given parameter in planet_sc_args was not given a bound'
    for i, dictvals in enumerate(ring_sc_args.items()):
        key = dictvals[0]
        item = dictvals[1]
        if key in ring_sc_args_bounds.keys():
            ring_sc_args_positional.append(item)
            ring_sc_bounds_positional.append(ring_sc_args_bounds[key])
            ring_sc_args_order[i] = key
        else:
            raise 'Given parameter in ring_sc_args was not given a bound'
    p0 = np.array([radius, inner_rad, ring_width, n_x, n_y, n_z, *planet_sc_args_positional, *ring_sc_args_positional])
    bounds = [radius_bounds, inner_rad_bounds, ring_width_bounds, n_x_bounds, n_y_bounds, n_z_bounds,
              *planet_sc_bounds_positional, *ring_sc_bounds_positional]
    result = op.minimize(
        lambda p: log_likelihood_ring(data, planet_sc_law, ring_sc_law, star, planet_sc_args_order, ring_sc_args_order,
                                      *p), p0, bounds=bounds).x
    output = dict()
    output['radius'], output['inner_rad'], output['ring_width'], n_x, n_y, n_z = result[:6]
    planet_sc_args = result[6:6 + len(planet_sc_args_positional)]
    ring_sc_args = result[6 + len(planet_sc_args_positional):]
    output['planet_sc_args'] = {planet_sc_args_order[index]: planet_sc_args[index] for index in
                                range(len(planet_sc_args))}
    output['ring_sc_args'] = {ring_sc_args_order[index]: ring_sc_args[index] for index in
                              range(len(ring_sc_args))}
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


# should turn fitting results into their own class later
def plot_planet_result(data, result_planetfit):
    plt.figure()
    plt.style.use('the_usual.mplstyle')
    fitted_planet = FittingPlanet(scattering.Jupiter, star, result_planetfit)
    alphas = np.linspace(-np.pi, np.pi, 10000)
    plt.errorbar(data[0], data[1], data[2], fmt='.')
    plt.plot(alphas, fitted_planet.light_curve(alphas))
    plt.savefig('images/PlanetFit',dpi=600)


def plot_ring_result(data, result_ringfit):
    plt.figure()
    plt.style.use('the_usual.mplstyle')
    result_ringfit['n_x'] = result_ringfit['ring_normal'][0]
    result_ringfit['n_y'] = result_ringfit['ring_normal'][1]
    result_ringfit['n_z'] = result_ringfit['ring_normal'][2]
    fitted_planet = FittingRingedPlanet(scattering.Jupiter, scattering.Rayleigh, star, result_ringfit)
    alphas = np.linspace(-np.pi, np.pi, 10000)
    plt.errorbar(data[0], data[1], data[2], fmt='.')
    plt.plot(alphas, fitted_planet.light_curve(alphas))
    plt.savefig('images/RingFit', dpi=600)


star = exoring_objects.Star(1, SUN_TO_JUP, 0.1 * AU_TO_JUP, 1)
test_planet = exoring_objects.Planet(scattering.Jupiter(0.9), 1, star)
test_ring_planet = exoring_objects.RingedPlanet(scattering.Jupiter(1), 1, scattering.Rayleigh(0.9), 2, 3,
                                                np.array([1., 1., 0]), star)
planet_data = generate_data(test_planet)
ring_data = generate_data(test_ring_planet)

init_guess_planet = {'radius': (1.4, (0, np.inf)), 'planet_sc_args': ({'albedo': 0.7}, {'albedo': (0, 1)})}
init_guess_ring = {'radius': (1, (0, np.inf)), 'inner_rad': (1, (1, np.inf)), 'ring_width': (0.5, (0, np.inf)),
                   'ring_normal': ([1, 1, 0], [(0, 1), (0, 1), (0, 1)]),
                   'planet_sc_args': ({'albedo': 1}, {'albedo': (0, 1)}),
                   'ring_sc_args': ({'albedo': 1}, {'albedo': (0, 1)})}

result_planetfit = fit_data_planet(planet_data, scattering.Jupiter, star, init_guess_planet)
#result_ringfit = fit_data_ring(planet_data, scattering.Jupiter, scattering.Rayleigh, star, init_guess_ring)
plot_planet_result(planet_data, result_planetfit)
#plot_ring_result(planet_data, result_ringfit)
print('planetfit', result_planetfit)
# print('ringfit', result_ringfit)

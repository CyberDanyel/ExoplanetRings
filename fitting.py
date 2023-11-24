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
from multiprocessing import Pool, freeze_support
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


class PerformFit():
    def __init__(self, data, star):
        self.data = data
        self.star = star

    def log_likelihood_ring(self, planet_sc_law, ring_sc_law, planet_sc_args_order, ring_sc_args_order, *params):
        alpha = self.data[0]
        I = self.data[1]
        I_errs = self.data[2]
        planet_sc_args_positional = params[6:6 + len(planet_sc_law.__init__.__code__.co_varnames) - 1]
        ring_sc_args_positional = params[6 + len(planet_sc_law.__init__.__code__.co_varnames) - 1:]
        parameters = {'radius': params[0], 'inner_rad': params[1], 'ring_width': params[2], 'n_x': params[3],
                      'n_y': params[4], 'n_z': params[5],
                      'planet_sc_args': {planet_sc_args_order[index]: planet_sc_args_positional[index] for index in
                                         range(len(planet_sc_args_positional))},
                      'ring_sc_args': {ring_sc_args_order[index]: ring_sc_args_positional[index] for index in
                                       range(len(ring_sc_args_positional))}}
        model_ringed_planet = FittingRingedPlanet(planet_sc_law, ring_sc_law, self.star, parameters) # problem occurs here
        x = model_ringed_planet.light_curve(alpha)
        with np.errstate(divide='raise'):
            try:
                # print(-np.sum(np.log(gaussian(x, I, I_errs))))
                return -np.sum(np.log(gaussian(x, I, I_errs)))
            except:  # The gaussian has returned 0 for at least 1 data point
                with np.errstate(divide='ignore'):
                    # print('Triggered')
                    logs = np.log(gaussian(x, I, I_errs))
                    for index, element in enumerate(logs):
                        if np.isinf(element):
                            logs[index] = -1000
                    # print(-np.sum(logs))
                    return -np.sum(logs)

    def log_likelihood_planet(self, sc_law, planet_sc_args_order, *params):
        alpha = self.data[0]
        I = self.data[1]
        I_errs = self.data[2]
        planet_sc_args_positional = params[1:]
        parameters = {'radius': params[0],
                      'planet_sc_args': {planet_sc_args_order[index]: planet_sc_args_positional[index] for index in
                                         range(len(planet_sc_args_positional))}}
        model_planet = FittingPlanet(sc_law, self.star, parameters)
        x = model_planet.light_curve(alpha)
        with np.errstate(divide='raise'):
            try:
                # print(-np.sum(np.log(gaussian(x, I, I_errs))))
                return -np.sum(np.log(gaussian(x, I, I_errs)))
            except:  # The gaussian has returned 0 for at least 1 data point
                with np.errstate(divide='ignore'):
                    # print('Triggered')
                    logs = np.log(gaussian(x, I, I_errs))
                    for index, element in enumerate(logs):
                        if np.isinf(element):
                            logs[index] = -1000
                    # print(-np.sum(logs))
                    return -np.sum(logs)

    def fit_data_planet(self, planet_sc_law, init_guess: dict):
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
        minimization = op.minimize(
            lambda p: self.log_likelihood_planet(planet_sc_law, planet_sc_args_order, *p), p0,
            bounds=bounds, method='COBYLA')
        result = minimization.x
        NLL = minimization.fun
        success = minimization.success
        output = dict()
        output['radius'] = result[0]
        planet_sc_args = result[1:]
        output['planet_sc_args'] = {planet_sc_args_order[index]: planet_sc_args[index] for index in
                                    range(len(planet_sc_args))}
        return NLL  # change this

    def fitwrap_planet(self, args):
        planet_sc_law = args[0]
        init_guess_planet = args[1]
        return self.fit_data_planet(planet_sc_law, init_guess_planet)

    def fit_data_ring(self, planet_sc_law, ring_sc_law, init_guess: dict):
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
        p0 = np.array(
            [radius, inner_rad, ring_width, n_x, n_y, n_z, *planet_sc_args_positional, *ring_sc_args_positional])
        bounds = [radius_bounds, inner_rad_bounds, ring_width_bounds, n_x_bounds, n_y_bounds, n_z_bounds,
                  *planet_sc_bounds_positional, *ring_sc_bounds_positional]
        minimization = op.minimize(
            lambda p: self.log_likelihood_ring(planet_sc_law, ring_sc_law, planet_sc_args_order, ring_sc_args_order,
                                               *p), p0, bounds=bounds, method='COBYLA')
        result = minimization.x
        NLL = minimization.fun
        success = minimization.success
        output = dict()
        output['radius'], output['inner_rad'], output['ring_width'], n_x, n_y, n_z = result[:6]
        planet_sc_args = result[6:6 + len(planet_sc_args_positional)]
        ring_sc_args = result[6 + len(planet_sc_args_positional):]
        output['planet_sc_args'] = {planet_sc_args_order[index]: planet_sc_args[index] for index in
                                    range(len(planet_sc_args))}
        output['ring_sc_args'] = {ring_sc_args_order[index]: ring_sc_args[index] for index in
                                  range(len(ring_sc_args))}
        output['ring_normal'] = (n_x, n_y, n_z)
        return NLL

    def fitwrap_ring(self, args):
        planet_sc_law = args[0]
        ring_sc_law = args[1]
        init_guess_ring = args[2]
        return self.fit_data_ring(planet_sc_law, ring_sc_law, init_guess_ring)

    # should turn fitting results into their own class later
    def plot_planet_result(self, result_planetfit):
        plt.figure()
        plt.style.use('the_usual.mplstyle')
        fitted_planet = FittingPlanet(scattering.Jupiter, self.star, result_planetfit)
        alphas = np.linspace(-np.pi, np.pi, 10000)
        plt.errorbar(self.data[0], self.data[1], self.data[2], fmt='.')
        plt.plot(alphas, fitted_planet.light_curve(alphas))
        plt.savefig('images/PlanetFit', dpi=600)

    def plot_ring_result(self, result_ringfit):
        plt.figure()
        plt.style.use('the_usual.mplstyle')
        result_ringfit['n_x'] = result_ringfit['ring_normal'][0]
        result_ringfit['n_y'] = result_ringfit['ring_normal'][1]
        result_ringfit['n_z'] = result_ringfit['ring_normal'][2]
        fitted_planet = FittingRingedPlanet(scattering.Jupiter, scattering.Rayleigh, self.star, result_ringfit)
        alphas = np.linspace(-np.pi, np.pi, 10000)
        plt.errorbar(self.data[0], self.data[1], self.data[2], fmt='.')
        plt.plot(alphas, fitted_planet.light_curve(alphas))
        plt.savefig('images/RingFit', dpi=600)

    def assign_bounds(self, bounds: dict, boundless_init_guess: dict):
        init_guess = dict()
        for key in boundless_init_guess:
            if key in bounds.keys():
                init_guess[key] = (boundless_init_guess[key], bounds[key])
            else:
                raise f'Bound not given for {key}'
        return init_guess

    def perform_fitting(self, search_ranges, search_interval, bounds, planet_sc_functions, ring_sc_functions=None,
                        search_values=None):
        if not search_values:
            search_ranges['n_x'] = search_ranges['ring_normal'][0]
            search_ranges['n_y'] = search_ranges['ring_normal'][1]
            search_ranges['n_z'] = search_ranges['ring_normal'][2]
            if ring_sc_functions:
                try:
                    del search_ranges['ring_normal']
                except KeyError:
                    raise 'No ring_normal given'
            search_values = list()
            parameter_ordering = dict()
            for order, key in enumerate(search_ranges):
                if key == 'planet_sc_args' or key == 'ring_sc_args':
                    for sc_arg in search_ranges[key].keys():
                        search_values.append([search_ranges[key][sc_arg][0] + (
                                (search_ranges[key][sc_arg][1] - search_ranges[key][sc_arg][0]) / (
                                1 / search_interval)) * i for i in range(int(1 / search_interval))])
                        parameter_ordering[order] = key + ',' + sc_arg
                else:
                    search_values.append([search_ranges[key][0] + (
                            (search_ranges[key][1] - search_ranges[key][0]) / (1 / search_interval)) * i for i in
                                          range(int(1 / search_interval))])
                    parameter_ordering[order] = key
        grid = np.meshgrid(*search_values)
        positions = np.vstack(list(map(np.ravel, grid)))
        init_guesses = list()
        for iteration in range(len(positions[0])):
            search_positions = dict()
            for order in range(len(positions)):
                if ',' not in parameter_ordering[order]:
                    search_positions[parameter_ordering[order]] = positions[order][iteration]
                else:
                    try:
                        search_positions[parameter_ordering[order].split(',')[0]][
                            parameter_ordering[order].split(',')[1]] = \
                            positions[order][iteration]
                    except KeyError:
                        search_positions[parameter_ordering[order].split(',')[0]] = dict()
                        search_positions[parameter_ordering[order].split(',')[0]][
                            parameter_ordering[order].split(',')[1]] = \
                            positions[order][iteration]
            search_positions['ring_normal'] = [search_positions['n_x'], search_positions['n_y'],
                                               search_positions['n_z']]
            if ring_sc_functions:
                del search_positions['n_x'],search_positions['n_y'],search_positions['n_z']
            search_positions = self.assign_bounds(bounds, search_positions)
            init_guesses.append(search_positions)
        freeze_support()
        res = list()
        if ring_sc_functions:
            #with Pool(processes=len(positions[0])) as pool:
            #with Pool(16) as pool:
            for planet_sc in planet_sc_functions:
                for ring_sc in ring_sc_functions:
                    args = [(planet_sc, ring_sc, guess) for guess in init_guesses]
                    for arg in args:
                        res.append(self.fitwrap_ring(arg))
                        return res
        else:
            #with Pool(processes=len(positions[0])) as pool:
            #with Pool(16) as pool:
            for planet_sc in planet_sc_functions:
                args = [(planet_sc, guess) for guess in init_guesses]
                return pool.map(self.fitwrap_planet, args)


def generate_data(test_planet):
    test_alphas = list(np.linspace(-np.pi, -.3, 10)) + list(np.linspace(-.29, .29, 10)) + list(
        np.linspace(.3, np.pi, 10))
    test_alphas = np.array(test_alphas)
    I = test_planet.light_curve(test_alphas)
    errs = 0.02 * I + 1e-8
    noise_vals = np.random.normal(size=len(test_alphas))
    data_vals = errs * noise_vals + I
    data = np.array([test_alphas, data_vals, errs])
    return data

'''
star_obj = exoring_objects.Star(1, SUN_TO_JUP, 0.1 * AU_TO_JUP, 1)
test_planet = exoring_objects.Planet(scattering.Jupiter(0.9), 1, star_obj)
test_ring_planet = exoring_objects.RingedPlanet(scattering.Jupiter(1), 1, scattering.Rayleigh(0.9), 2, 3,
                                                np.array([1., 1., 0]), star_obj)
planet_data = generate_data(test_planet)
ring_data = generate_data(test_ring_planet)

init_guess_planet = {'radius': (4, (0, np.inf)), 'planet_sc_args': ({'albedo': 0.3}, {'albedo': (0, 1)})}

init_guess_ring = {'radius': (1, (0, np.inf)), 'inner_rad': (1, (1, np.inf)), 'ring_width': (0.5, (0, np.inf)),
                   'ring_normal': ([1, 1, 0], [(0, 1), (0, 1), (0, 1)]),
                   'planet_sc_args': ({'albedo': 1}, {'albedo': (0, 1)}),
                   'ring_sc_args': ({'albedo': 1}, {'albedo': (0, 1)})}

search_ranges_planet = {'radius': (1, 4), 'planet_sc_args': {'albedo': (0, 1)}}
search_ranges_ring = {'radius': (0, 2), 'inner_rad': (1, 3), 'ring_width': (1, 2),
                      'ring_normal': [(0, 1), (0, 1), (0, 1)],
                      'planet_sc_args': {'albedo': (0, 1)},
                      'ring_sc_args': {'albedo': (0, 1)}}
bounds_ring = {'radius': (0, np.inf), 'inner_rad': (0, np.inf), 'ring_width': (0.1, np.inf),
               'ring_normal': [(0, 1), (0, 1), (0, 1)],
               'planet_sc_args': {'albedo': (0, 1)},
               'ring_sc_args': {'albedo': (0, 1)}}
Fit = PerformFit(ring_data, star_obj)
# result_planetfit = Fit.fit_data_planet(scattering.Jupiter, init_guess_planet)
result = Fit.perform_fitting(search_ranges_ring, 0.2, bounds_ring, scattering.Jupiter, scattering.Rayleigh)
# result_ringfit = fit_data_ring(planet_data, scattering.Jupiter, scattering.Rayleigh, star, init_guess_ring)
# Fit.plot_planet_result(result_planetfit[0])
# plot_ring_result(planet_data, result_ringfit)
# print('planetfit', result_planetfit[0])
'''
# print('ringfit', result_ringfit)

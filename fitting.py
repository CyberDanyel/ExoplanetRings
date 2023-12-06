import json

import numpy as np
import matplotlib.pyplot as plt

import exoring_functions
import exoring_objects
import scattering
import scipy.optimize as op
import time
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
        disk_gap, ring_width, n_x, n_y, n_z, ring_sc_args = parameters['disk_gap'], parameters['ring_width'], \
            parameters['n_x'], parameters['n_y'], parameters['n_z'], parameters['ring_sc_args']
        try:
            ring_sc = ring_sc_law(**ring_sc_args)
        except TypeError:
            raise TypeError('Too little/many arguments for ring sc law')

        exoring_objects.RingedPlanet.__init__(self, self.sc_law, self.radius, ring_sc, self.radius + disk_gap,
                                              self.radius + disk_gap + ring_width, [n_x, n_y, n_z], star)
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
        self.best_result_ring = None
        self.best_result_planet = None
        self.data = data
        self.star = star

    def log_likelihood_planet(self, sc_law, planet_sc_args_order, *params):
        # Calculates log likelihood for specific planet params and scattering law
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

    def log_likelihood_ring(self, planet_sc_law, ring_sc_law, planet_sc_args_order, ring_sc_args_order, *params):
        # Calculates log likelihood for specific ringed planet params and scattering laws
        alpha = self.data[0]
        I = self.data[1]
        I_errs = self.data[2]
        planet_sc_args_positional = params[6:6 + len(planet_sc_law.__init__.__code__.co_varnames) - 1]
        ring_sc_args_positional = params[6 + len(planet_sc_law.__init__.__code__.co_varnames) - 1:]
        parameters = {'radius': params[0], 'disk_gap': params[1], 'ring_width': params[2], 'n_x': params[3],
                      'n_y': params[4], 'n_z': params[5],
                      'planet_sc_args': {planet_sc_args_order[index]: planet_sc_args_positional[index] for index in
                                         range(len(planet_sc_args_positional))},
                      'ring_sc_args': {ring_sc_args_order[index]: ring_sc_args_positional[index] for index in
                                       range(len(ring_sc_args_positional))}}
        if parameters['n_x'] < 0:
            print('n_x was inputted to be <0, possibly the minimiser not respecting bounds')
        model_ringed_planet = FittingRingedPlanet(planet_sc_law, ring_sc_law, self.star, parameters)
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

    def fit_data_planet(self, planet_sc_law, init_guess: dict):
        """
        Finds planet giving best fit to the data with given planet scattering law and initial guesses

        Parameters
        ----------
        planet_sc_law: planet scattering law class
        init_guess: initial guess for minimisation

        Returns
        -------
        result of minimisation
        NLL of minimisation
        success of minimisation

        """
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
            bounds=bounds)
        result = minimization.x
        NLL = minimization.fun
        success = minimization.success
        if not success:
            return np.inf, None
        elif success:
            output = dict()
            output['radius'] = result[0]
            planet_sc_args = result[1:]
            output['planet_sc_args'] = {planet_sc_args_order[index]: planet_sc_args[index] for index in
                                        range(len(planet_sc_args))}
            return NLL, output

    def fitwrap_planet(self, args):
        # pool.map only takes a function with a single argument, so all arguments are wrapped here and parsed onto
        # the fitting function
        planet_sc_law = args[0]
        init_guess_planet = args[1]
        return self.fit_data_planet(planet_sc_law, init_guess_planet)

    def fit_data_ring(self, planet_sc_law, ring_sc_law, init_guess: dict):
        """
        Finds ringed planet giving best fit to the data with given planet & ring scattering laws and initial guesses

        Parameters
        ----------
        planet_sc_law: planet scattering law class
        ring_sc_law: ring scattering law class
        init_guess: initial guess for minimisation

        Returns
        -------
        result of minimisation
        NLL of minimisation
        success of minimisation

        """
        try:
            radius = init_guess['radius'][0]
            radius_bounds = init_guess['radius'][1]
            disk_gap = init_guess['disk_gap'][0]
            disk_gap_bounds = init_guess['disk_gap'][1]
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
            [radius, disk_gap, ring_width, n_x, n_y, n_z, *planet_sc_args_positional, *ring_sc_args_positional])
        bounds = [radius_bounds, disk_gap_bounds, ring_width_bounds, n_x_bounds, n_y_bounds, n_z_bounds,
                  *planet_sc_bounds_positional, *ring_sc_bounds_positional]
        minimization = op.minimize(
            lambda p: self.log_likelihood_ring(planet_sc_law, ring_sc_law, planet_sc_args_order, ring_sc_args_order,
                                               *p), p0, bounds=bounds)
        result = minimization.x
        NLL = minimization.fun
        success = minimization.success
        if not success:
            return np.inf, None
        elif success:
            output = dict()
            output['radius'], output['disk_gap'], output['ring_width'], n_x, n_y, n_z = result[:6]
            planet_sc_args = result[6:6 + len(planet_sc_args_positional)]
            ring_sc_args = result[6 + len(planet_sc_args_positional):]
            output['planet_sc_args'] = {planet_sc_args_order[index]: planet_sc_args[index] for index in
                                        range(len(planet_sc_args))}
            output['ring_sc_args'] = {ring_sc_args_order[index]: ring_sc_args[index] for index in
                                      range(len(ring_sc_args))}
            ring_normal = np.array([n_x, n_y, n_z])
            ring_normal /= np.linalg.norm(ring_normal)
            output['ring_normal'] = tuple(ring_normal)
            return NLL, output

    def fitwrap_ring(self, args):
        # pool.map only takes a function with a single argument, so all arguments are wrapped here and parsed onto
        # the fitting function
        planet_sc_law = args[0]
        ring_sc_law = args[1]
        init_guess_ring = args[2]
        return self.fit_data_ring(planet_sc_law, ring_sc_law, init_guess_ring)

    # should turn fitting results into their own class later
    def plot_planet_result(self, result_planetfit, planet_sc):
        fig, ax = exoring_functions.generate_plot_style()
        plt.style.use('the_usual.mplstyle')
        fitted_planet = FittingPlanet(planet_sc, self.star, result_planetfit)
        alphas = np.linspace(-np.pi, np.pi, 10000)
        ax.errorbar(self.data[0] / np.pi, self.data[1], self.data[2], fmt='.')
        ax.plot(alphas / np.pi, fitted_planet.light_curve(alphas))
        plt.savefig('images/PlanetFit', dpi=600)

    def plot_ring_result(self, result_ringfit, planet_sc, ring_sc):
        fig, ax = exoring_functions.generate_plot_style()
        plt.style.use('the_usual.mplstyle')
        result_ringfit['n_x'] = result_ringfit['ring_normal'][0]
        result_ringfit['n_y'] = result_ringfit['ring_normal'][1]
        result_ringfit['n_z'] = result_ringfit['ring_normal'][2]
        fitted_planet = FittingRingedPlanet(planet_sc, ring_sc, self.star, result_ringfit)
        alphas = np.linspace(-np.pi, np.pi, 10000)
        ax.errorbar(self.data[0] / np.pi, self.data[1], self.data[2], fmt='.')
        ax.plot(alphas / np.pi, fitted_planet.light_curve(alphas))
        plt.savefig('images/BestRingFit', dpi=600)

    def assign_bounds(self, bounds: dict, boundless_init_guess: dict):
        # fit_data_planet and fit_data_ring take bounds in the initial guesses.
        # This assigns given bounds to a boundless initial guess to be used in the perform_fitting function
        init_guess = dict()
        for key in boundless_init_guess:
            if key in bounds.keys():
                init_guess[key] = (boundless_init_guess[key], bounds[key])
            else:
                raise f'Bound not given for {key}'
        return init_guess

    def perform_fitting(self, search_ranges, search_interval, bounds, planet_sc_functions, ring_sc_functions=None,
                        search_values=None):
        # Creates meshgrid of initial_values to form multiple initial guesses. Runs minimisation for all these
        # initial guesses with every possible combination of planet & ring scattering functions If you want the data
        # to be optimised with a ringed planet model, include ring_sc_functions. If you want a planet-only fit,
        # leave it blank.

        """

        Parameters
        ----------
        search_ranges: dictionary with desired search ranges for initial values
        search_interval: from 0 to 1, fractional difference seeked between initial values created.
                    Search ranges (0,1) with search_interval = 0.2 will create 5 initial values evenly spaced from 0 to 1
        bounds: bounds for minimisation
        planet_sc_functions: planet scattering function
        ring_sc_functions: ring scattering function
        search_values: if you want to use specific initial values in the search, use this (currently not implemented)

        Returns
        -------
        Whatever you want it to return. Maybe the initial guess with the lowest NLL if you rly want that
        what else do you want from me. Look man my shift ended 5 minutes ago if you have anymore questions
        you're gonna have to come tomorrow
        """
        if not search_values:
            if ring_sc_functions:
                try:
                    search_ranges['n_x'] = search_ranges['ring_normal'][0]
                    search_ranges['n_y'] = search_ranges['ring_normal'][1]
                    search_ranges['n_z'] = search_ranges['ring_normal'][2]
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
            if ring_sc_functions:
                search_positions['ring_normal'] = [search_positions['n_x'], search_positions['n_y'],
                                                   search_positions['n_z']]
                del search_positions['n_x'], search_positions['n_y'], search_positions['n_z']
            search_positions = self.assign_bounds(bounds, search_positions)
            init_guesses.append(search_positions)
        freeze_support()
        lowest_NLL = np.inf
        if ring_sc_functions:
            with Pool(16) as pool:
                for planet_sc in planet_sc_functions:
                    for ring_sc in ring_sc_functions:
                        args = [(planet_sc, ring_sc, guess) for guess in init_guesses]
                        results = pool.map(self.fitwrap_ring, args)
                        best_NLL, best_fit = exoring_functions.select_best_result(results)
                        if best_NLL < lowest_NLL:
                            best_result = (best_NLL, best_fit, planet_sc, ring_sc)
                            lowest_NLL = best_NLL
                        else:
                            pass
                self.best_result_ring = best_result
                json_serializable_best_result = {'NLL': best_result[0],
                                                 'Result': best_result[1],
                                                 'Planet_sc_function': best_result[2].__name__,
                                                 'Ring_sc_function': best_result[3].__name__}
                with open('best_fit_ring.json', 'w') as f:
                    json.dump(json_serializable_best_result, f)

        else:
            with Pool(16) as pool:
                for planet_sc in planet_sc_functions:
                    args = [(planet_sc, guess) for guess in init_guesses]
                    results = pool.map(self.fitwrap_planet, args)
                    best_NLL, best_fit = exoring_functions.select_best_result(results)
                    if best_NLL < lowest_NLL:
                        best_result = (best_NLL, best_fit, planet_sc)
                        lowest_NLL = best_NLL
                    else:
                        pass
                self.best_result_planet = best_result
                json_serializable_best_result = {'NLL': best_result[0],
                                                 'Result': best_result[1],
                                                 'Planet_sc_function': best_result[2].__name__}
                with open('best_fit_planet.json', 'w') as f:
                    json.dump(json_serializable_best_result, f)

    def plot_best_ringfit(self):
        self.plot_ring_result(self.best_result_ring[1], self.best_result_ring[2], self.best_result_ring[3])

    def plot_best_planetfit(self):
        self.plot_planet_result(self.best_result_planet[1], self.best_result_planet[2])


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

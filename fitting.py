import json
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as tck
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import AutoMinorLocator
import scipy.integrate as integrate

import exoring_functions
import exoring_objects
import scattering
import scipy.optimize as op
import time
from multiprocessing import Pool, freeze_support

primes_to_100 = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


# 2 not included because it does fit nicely on a graph
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
        return (1 / (sigma*np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


class Data_Object():
    def __init__(self, data, star):
        self.best_result_ring = None
        self.best_result_planet = None
        self.data = data
        self.star = star

    def log_likelihood_planet(self, sc_law, planet_sc_args_order, *params):  # used for fitting
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

    def log_likelihood_ring(self, planet_sc_law, ring_sc_law, planet_sc_args_order, ring_sc_args_order,
                            *params):  # used for fitting
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
    def plot_planet_result(self, result, planet_sc):
        plt.style.use('the_usual.mplstyle')
        fig, ax = exoring_functions.generate_plot_style()
        fitted_planet = FittingPlanet(planet_sc, self.star, result)
        alphas = np.linspace(-np.pi, np.pi, 10000)
        ax.errorbar(self.data[0] / np.pi, self.data[1], self.data[2], fmt='.')
        ax.plot(alphas / np.pi, fitted_planet.light_curve(alphas))
        plt.savefig('images/PlanetFit', dpi=600)

    def plot_ring_result(self, result, planet_sc, ring_sc):
        plt.style.use('the_usual.mplstyle')
        fig, ax = exoring_functions.generate_plot_style()
        result['n_x'] = result['ring_normal'][0]
        result['n_y'] = result['ring_normal'][1]
        result['n_z'] = result['ring_normal'][2]
        fitted_planet = FittingRingedPlanet(planet_sc, ring_sc, self.star, result)
        alphas = np.linspace(-np.pi, np.pi, 10000)
        ax.errorbar(self.data[0] / np.pi, self.data[1], self.data[2], fmt='.')
        ax.plot(alphas / np.pi, fitted_planet.light_curve(alphas))
        plt.savefig('images/RingFit', dpi=600)

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

    def range_search_fitting(self, search_ranges, search_interval, bounds, planet_sc_functions, ring_sc_functions=None):
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
                    json.dump(json_serializable_best_result, f, indent=4)

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
                    json.dump(json_serializable_best_result, f, indent=4)

    def run_ringless_model(self, planet_sc_law, model_parameters):
        plt.style.use('the_usual.mplstyle')
        fig, ax = exoring_functions.generate_plot_style()
        I = self.data[1]
        I_errs = self.data[2]
        if model_parameters['n_x'] < 0:
            print('n_x was inputted to be <0')
        model_ringless_planet = FittingPlanet(planet_sc_law, self.star, model_parameters)
        alphas = np.linspace(-np.pi, np.pi, 10000)
        resulting_lightcurve = model_ringless_planet.light_curve(alphas)
        ax.errorbar(self.data[0] / np.pi, I, I_errs, fmt='.')
        ax.plot(alphas / np.pi, resulting_lightcurve)
        plt.savefig('images/Ringless_Model', dpi=600)

    def run_ringed_model(self, planet_sc_law, ring_sc_law, model_parameters):
        plt.style.use('the_usual.mplstyle')
        fig, ax = exoring_functions.generate_plot_style()
        model_parameters['n_x'], model_parameters['n_y'], model_parameters['n_z'] = model_parameters['ring_normal'][0], \
            model_parameters['ring_normal'][1], model_parameters['ring_normal'][2]
        I = self.data[1]
        I_errs = self.data[2]
        if model_parameters['n_x'] < 0:
            print('n_x was inputted to be <0')
        model_ringed_planet = FittingRingedPlanet(planet_sc_law, ring_sc_law, self.star, model_parameters)
        alphas = np.linspace(-np.pi, np.pi, 10000)
        resulting_lightcurve = model_ringed_planet.light_curve(alphas)
        ax.errorbar(self.data[0] / np.pi, I, I_errs, fmt='.')
        ax.plot(alphas / np.pi, resulting_lightcurve)
        plt.savefig('images/Ringed_Model', dpi=600)

    def run_many_ringless_models(self, planet_sc_law, multiple_model_parameters, sharex=True, sharey=False):
        length = len(multiple_model_parameters)
        nrows = np.sqrt(length)
        if nrows.is_integer():
            ncols = nrows
            plt.style.use('the_usual.mplstyle')
            fig, axs = plt.subplots(int(nrows), int(ncols), sharex=sharex, sharey=sharey)
        else:
            if length not in primes_to_100:
                nrows = math.ceil(nrows)
                while True:
                    ncols = length / nrows
                    if ncols.is_integer():
                        nrows, ncols = min((nrows, ncols)), max((nrows, ncols))
                        plt.style.use('the_usual.mplstyle')
                        fig, axs = plt.subplots(int(nrows), int(ncols), sharex=sharex, sharey=sharey)
                        break
                    else:
                        nrows += 1
            elif length in primes_to_100:
                if length == 3:
                    plt.style.use('the_usual.mplstyle')
                    fig, axs = plt.subplots(1, 3, sharex=sharex, sharey=sharey)
                elif length == 5:
                    plt.style.use('the_usual.mplstyle')
                    fig, axs = plt.subplots(2, 3, sharex=sharex, sharey=sharey)
                elif length == 7:
                    plt.style.use('the_usual.mplstyle')
                    fig, axs = plt.subplots(3, 3, sharex=sharex, sharey=sharey)
                else:
                    raise NotImplementedError('This is too many models to fit nicely')
            else:
                raise NotImplementedError('Number is prime and too large')
        for model_parameters, ax in zip(multiple_model_parameters, axs.flat):
            I = self.data[1]
            I_errs = self.data[2]
            model_ringed_planet = FittingPlanet(planet_sc_law, self.star, model_parameters)
            alphas = np.linspace(-np.pi, np.pi, 10000)
            resulting_lightcurve = model_ringed_planet.light_curve(alphas)
            ax.errorbar(self.data[0] / np.pi, I, I_errs, fmt='.')
            ax.plot(alphas / np.pi, resulting_lightcurve)
            ax.set_title(
                f'R:{round(model_parameters['radius'])}', fontsize=8, pad=2)
        if int(nrows) == 1:
            for ax in axs:
                ax.xaxis.set_major_formatter(FuncFormatter(exoring_functions.format_fraction_with_pi))
                ax.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
                ax.set_xlabel(r'Phase angle $\alpha$')
                for row in range(int(nrows)):
                    if ax == axs[0]:  # If in the first column
                        ax.set_ylabel(r'Intensity ($L_{\odot}$)')
        else:
            for ax in axs.flat:
                ax.xaxis.set_major_formatter(FuncFormatter(exoring_functions.format_fraction_with_pi))
                ax.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
                if ax in axs[int(nrows) - 1]:  # If in the last row
                    ax.set_xlabel(r'Phase angle $\alpha$')
                for row in range(int(nrows)):
                    if ax == axs[row][0]:  # If in the first column
                        ax.set_ylabel(r'Intensity ($L_{\odot}$)')

        plt.savefig('images/Multiples', dpi=1000)

    def run_many_ringed_models(self, planet_sc_law, ring_sc_law, multiple_model_parameters, sharex=True, sharey=False):
        length = len(multiple_model_parameters)
        nrows = np.sqrt(length)
        if nrows.is_integer():
            ncols = nrows
            plt.style.use('the_usual.mplstyle')
            fig, axs = plt.subplots(int(nrows), int(ncols), sharex=sharex, sharey=sharey)
        else:
            if length not in primes_to_100:
                nrows = math.ceil(nrows)
                while True:
                    ncols = length / nrows
                    if ncols.is_integer():
                        nrows, ncols = min((nrows, ncols)), max((nrows, ncols))
                        plt.style.use('the_usual.mplstyle')
                        fig, axs = plt.subplots(int(nrows), int(ncols), sharex=sharex, sharey=sharey)
                        break
                    else:
                        nrows += 1
            elif length in primes_to_100:
                if length == 3:
                    plt.style.use('the_usual.mplstyle')
                    fig, axs = plt.subplots(1, 3, sharex=sharex, sharey=sharey)
                elif length == 5:
                    plt.style.use('the_usual.mplstyle')
                    fig, axs = plt.subplots(2, 3, sharex=sharex, sharey=sharey)
                elif length == 7:
                    plt.style.use('the_usual.mplstyle')
                    fig, axs = plt.subplots(3, 3, sharex=sharex, sharey=sharey)
                else:
                    raise NotImplementedError('This is too many models to fit nicely')
            else:
                raise NotImplementedError('Number is prime and too large')
        for model_parameters, ax in zip(multiple_model_parameters, axs.flat):
            I = self.data[1]
            I_errs = self.data[2]
            model_parameters['n_x'], model_parameters['n_y'], model_parameters['n_z'] = model_parameters['ring_normal'][
                0], model_parameters['ring_normal'][1], model_parameters['ring_normal'][2]
            model_ringed_planet = FittingRingedPlanet(planet_sc_law, ring_sc_law, self.star, model_parameters)
            alphas = np.linspace(-np.pi, np.pi, 10000)
            resulting_lightcurve = model_ringed_planet.light_curve(alphas)
            ax.errorbar(self.data[0] / np.pi, I, I_errs, fmt='.')
            ax.plot(alphas / np.pi, resulting_lightcurve)
            ax.set_title(
                f'R:{round(model_parameters['radius'])} S:{round(model_parameters['disk_gap'])} W:{round(model_parameters['ring_width'])}',
                fontsize=8, pad=2)
        if int(nrows) == 1:
            for ax in axs:
                ax.xaxis.set_major_formatter(FuncFormatter(exoring_functions.format_fraction_with_pi))
                ax.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
                ax.set_xlabel(r'Phase angle $\alpha$')
                for row in range(int(nrows)):
                    if ax == axs[0]:  # If in the first column
                        ax.set_ylabel(r'Intensity ($L_{\odot}$)')
        else:
            for ax in axs.flat:
                ax.xaxis.set_major_formatter(FuncFormatter(exoring_functions.format_fraction_with_pi))
                ax.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
                if ax in axs[int(nrows) - 1]:  # If in the last row
                    ax.set_xlabel(r'Phase angle $\alpha$')
                for row in range(int(nrows)):
                    if ax == axs[row][0]:  # If in the first column
                        ax.set_ylabel(r'Intensity ($L_{\odot}$)')

        plt.savefig('images/Multiples', dpi=1000)

    def create_various_model_parameters(self, **kwargs):
        # Changing albedos not implemented
        all_dicts = list()
        all_params = list()
        order_dictionary = dict()
        default = {'radius': 1,
                   'disk_gap': 0.01, 'ring_width': 1,
                   'ring_normal': np.array([1., 1., 0]),
                   'planet_sc_args': {'albedo': 1},
                   'ring_sc_args': {'albedo': 0.01}}
        for order, key in enumerate(kwargs):
            param_list = list()
            param_list.append(kwargs[key])
            all_params.append(param_list)
            order_dictionary[order] = key
        grid = np.meshgrid(*all_params)
        positions = np.vstack(list(map(np.ravel, grid)))
        for iteration in range(len(positions[0])):
            for order in range(len(positions)):
                key = order_dictionary[order]
                new_dict = default.copy()
                new_dict[key] = positions[order][iteration]
            all_dicts.append(new_dict)
        return all_dicts

    def plot_best_ringfit(self):
        self.plot_ring_result(self.best_result_ring[1], self.best_result_ring[2], self.best_result_ring[3])

    def plot_best_planetfit(self):
        self.plot_planet_result(self.best_result_planet[1], self.best_result_planet[2])

    def likelihood_ringless_model(self, sc_law, model_parameters):  # Used for manual models
        # Calculates log likelihood for specific planet params and scattering law
        alpha = self.data[0]
        I = self.data[1]
        I_errs = self.data[2]
        model_planet = FittingPlanet(sc_law, self.star, model_parameters)
        x = model_planet.light_curve(alpha)
        return np.prod(gaussian(x, I, I_errs))

    def likelihood_ringed_model(self, planet_sc_law, ring_sc_law, model_parameters):  # Used for manual models
        # Calculates log likelihood for specific ringed planet params and scattering laws
        alpha = self.data[0]
        I = self.data[1]
        I_errs = self.data[2]
        model_parameters['n_x'], model_parameters['n_y'], model_parameters['n_z'] = model_parameters['ring_normal'][0], \
            model_parameters['ring_normal'][1], model_parameters['ring_normal'][2]
        if model_parameters['n_x'] < 0:
            print('n_x was inputted to be <0, possibly the minimiser not respecting bounds')
        model_ringed_planet = FittingRingedPlanet(planet_sc_law, ring_sc_law, self.star, model_parameters)
        x = model_ringed_planet.light_curve(alpha)
        return np.prod(gaussian(x, I, I_errs))

    def log_likelihood_ringless_model(self, sc_law, model_parameters):  # Used for manual models
        # Calculates log likelihood for specific planet params and scattering law
        alpha = self.data[0]
        I = self.data[1]
        I_errs = self.data[2]
        model_planet = FittingPlanet(sc_law, self.star, model_parameters)
        x = model_planet.light_curve(alpha)
        with np.errstate(divide='raise'):
            try:
                # print(-np.sum(np.log(gaussian(x, I, I_errs))))
                return np.sum(np.log(gaussian(x, I, I_errs)))
            except:  # The gaussian has returned 0 for at least 1 data point
                with np.errstate(divide='ignore'):
                    # print('Triggered')
                    logs = np.log(gaussian(x, I, I_errs))
                    for index, element in enumerate(logs):
                        if np.isinf(element):
                            logs[index] = -1000
                    # print(-np.sum(logs))
                    return np.sum(logs)

    def log_likelihood_ringed_model(self, planet_sc_law, ring_sc_law, model_parameters):  # Used for manual models
        # Calculates log likelihood for specific ringed planet params and scattering laws
        alpha = self.data[0]
        I = self.data[1]
        I_errs = self.data[2]
        model_parameters['n_x'], model_parameters['n_y'], model_parameters['n_z'] = model_parameters['ring_normal'][0], \
            model_parameters['ring_normal'][1], model_parameters['ring_normal'][2]
        if model_parameters['n_x'] < 0:
            print('n_x was inputted to be <0, possibly the minimiser not respecting bounds')
        model_ringed_planet = FittingRingedPlanet(planet_sc_law, ring_sc_law, self.star, model_parameters)
        x = model_ringed_planet.light_curve(alpha)
        with np.errstate(divide='raise'):
            try:
                # print(-np.sum(np.log(gaussian(x, I, I_errs))))
                return np.sum(np.log(gaussian(x, I, I_errs)))
            except:  # The gaussian has returned 0 for at least 1 data point
                with np.errstate(divide='ignore'):
                    # print('Triggered')
                    logs = np.log(gaussian(x, I, I_errs))
                    for index, element in enumerate(logs):
                        if np.isinf(element):
                            logs[index] = -1000
                    # print(-np.sum(logs))
                    return np.sum(logs)

    def produce_corner_plot(self, best_model, ranges, ringed, **kwargs):
        planet_sc_law = kwargs['planet_sc_law']
        if ringed:
            ring_sc_law = kwargs['ring_sc_law']
            best_ll = self.log_likelihood_ringed_model(planet_sc_law, ring_sc_law, best_model)
        else:
            best_ll = self.log_likelihood_ringless_model(planet_sc_law, best_model)
        keys = ranges.keys()
        keyslist = list(keys)
        all_params = list()
        key1 = keyslist[0]  # Temporary
        key2 = keyslist[1]
        key3 = keyslist[2]
        for key in keys:
            key_value_range = ranges[key]
            if key == key1:
                key_values = np.linspace(key_value_range[0], key_value_range[1], 2)
            if key == key2:
                key_values = np.linspace(key_value_range[0], key_value_range[1], 4)
            if key == key3:
                key_values = np.linspace(key_value_range[0], key_value_range[1], 6)
            all_params.append(key_values)
        X, Y, Z = np.meshgrid(*all_params)
        XsYs = np.meshgrid(all_params[0],all_params[1])
        XsZs = np.meshgrid(all_params[0], all_params[2])
        YsZs = np.meshgrid(all_params[1], all_params[2])
        X = np.swapaxes(X, 0, 1)
        Y = np.swapaxes(Y, 0, 1)
        Z = np.swapaxes(Z, 0, 1)
        with open('old_X.json', 'w') as f:
            json.dump(X.tolist(), f, indent=4)
        with open('old_Y.json', 'w') as f:
            json.dump(Y.tolist(), f, indent=4)
        with open('old_Z.json', 'w') as f:
            json.dump(Z.tolist(), f, indent=4)
        likelihood = np.zeros((len(X), len(X[0]), len(X[0][0])))
        for index_1 in range(len(X)):
            for index_2 in range(len(X)):
                for index_3 in range(len(X)):
                    altered_model = best_model.copy()
                    altered_model[key1] = X[index_1][index_2][index_3]
                    altered_model[key2] = Y[index_1][index_2][index_3]
                    altered_model[key3] = Z[index_1][index_2][index_3]
                    if ringed:
                        if altered_model['radius'] == 1.5 and altered_model['disk_gap'] == 0.01 and altered_model['ring_width'] == 0.01:
                            print('here')
                        likelihood_val = self.likelihood_ringed_model(planet_sc_law, ring_sc_law, altered_model)
                        likelihood[index_1][index_2][index_3] = likelihood_val
                    else:
                        likelihood_val = self.likelihood_ringless_model(planet_sc_law, altered_model)
                        likelihood[index_1][index_2][index_3] = likelihood_val
        with open('likelihood_old.json', 'w') as f:
            json.dump(likelihood.tolist(), f, indent=4)
        integral_over_Z = np.zeros((len(X),
                                    len(X[0])))
        for index_1 in range(len(X)):
            for index_2 in range(len(X[0])):
                val = np.trapz(likelihood[index_1][index_2], x=np.array(range(len(X[0][0]))))
                integral_over_Z[index_1][index_2] = val

        integral_over_Y = np.zeros(len(X))
        for index_1 in range(len(X)):
            val = np.trapz(integral_over_Z[index_1], x=np.array(range(len(X[0]))))
            integral_over_Y[index_1] = val

        total_integral = np.trapz(integral_over_Y, x=np.array(range(len(X))))
        if total_integral != 0:
            likelihood = likelihood / total_integral
        XY_contour_vals = np.zeros((len(X),
                                    len(X[0])))
        XZ_contour_vals = np.zeros((len(X),
                                    len(X[0][0])))
        YZ_contour_vals = np.zeros((len(X[0]),
                                    len(X[0][0])))
        for index_1 in range(len(X)):
            for index_2 in range(len(X[0])):
                val = np.trapz(likelihood[index_1][index_2], x=np.array(range(len(X[0][0]))))
                XY_contour_vals[index_1][index_2] = val
        likelihood_1_3_2 = np.swapaxes(likelihood, 1, 2)
        for index_1 in range(len(X)):
            for index_3 in range(len(X[0][0])):
                val = np.trapz(likelihood_1_3_2[index_1][index_3], x=np.array(range(len(X[0]))))
                XZ_contour_vals[index_1][index_3] = val
        likelihood_2_3_1 = np.swapaxes(likelihood_1_3_2, 2, 0)
        for index_2 in range(len(X[0])):
            for index_3 in range(len(X[0][0])):
                val = np.trapz(likelihood_2_3_1[index_2][index_3], x=np.array(range(len(X))))
                YZ_contour_vals[index_2][index_3] = val
        step = 0.01
        levels = np.arange(start=0, stop=XY_contour_vals.max()+step, step=step)
        plt.style.use('the_usual.mplstyle')
        fig, ax = plt.subplots()
        cp = ax.contourf(np.swapaxes(XsYs[0], 0, 1), np.swapaxes(XsYs[1], 0, 1), XY_contour_vals, levels, cmap='viridis')
        cbar = fig.colorbar(cp)  # Add a colorbar to a plot
        cbar.ax.tick_params(labelsize=12)
        ax.set_title(f'likelihood {key2} against {key1}', fontsize=13)
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
        ax.tick_params(direction='in', top=True, right=True, which='both', labelsize=12)
        # ax.set_xticks([0.21, 0.23, 0.25, 0.27, 0.29])
        # ax.set_yticks([1.95e-3, 2e-3, 2.05e-3, 2.1e-3, 2.15e-3, 2.2e-3, 2.25e-3])
        ax.set_xlabel(f'{key1}', fontsize=13)
        ax.set_ylabel(f'{key2}', fontsize=13)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.savefig(f'images/contour {key1}+{key2}', dpi=600)
        plt.show()

        levels = np.arange(start=0, stop=XZ_contour_vals.max()+step, step=step)
        plt.style.use('the_usual.mplstyle')
        fig, ax = plt.subplots()
        cp = ax.contourf(np.swapaxes(XsZs[0], 0, 1), np.swapaxes(XsZs[1], 0, 1), XZ_contour_vals, cmap='viridis', levels=levels)
        cbar = fig.colorbar(cp)  # Add a colorbar to a plot
        cbar.ax.tick_params(labelsize=12)
        ax.set_title(f'likelihood {key3} against {key1}', fontsize=13)
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
        ax.tick_params(direction='in', top=True, right=True, which='both', labelsize=12)
        # ax.set_xticks([0.21, 0.23, 0.25, 0.27, 0.29])
        # ax.set_yticks([1.95e-3, 2e-3, 2.05e-3, 2.1e-3, 2.15e-3, 2.2e-3, 2.25e-3])
        ax.set_xlabel(f'{key1}', fontsize=13)
        ax.set_ylabel(f'{key3}', fontsize=13)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.savefig(f'images/contour {key1}+{key3}', dpi=600)
        plt.show()

        levels = np.arange(start=0, stop=YZ_contour_vals.max()+step, step=step)
        plt.style.use('the_usual.mplstyle')
        fig, ax = plt.subplots()
        cp = ax.contourf(np.swapaxes(YsZs[0], 0, 1), np.swapaxes(YsZs[1], 0, 1), YZ_contour_vals, cmap='viridis', levels=levels)
        cbar = fig.colorbar(cp)  # Add a colorbar to a plot
        cbar.ax.tick_params(labelsize=12)
        ax.set_title(f'likelihood {key3} against {key2}', fontsize=13)
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
        ax.tick_params(direction='in', top=True, right=True, which='both', labelsize=12)
        # ax.set_xticks([0.21, 0.23, 0.25, 0.27, 0.29])
        # ax.set_yticks([1.95e-3, 2e-3, 2.05e-3, 2.1e-3, 2.15e-3, 2.2e-3, 2.25e-3])
        ax.set_xlabel(f'{key2}', fontsize=13)
        ax.set_ylabel(f'{key3}', fontsize=13)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.savefig(f'images/contour {key2}+{key3}', dpi=600)
        plt.show()


def generate_data(test_planet):
    np.random.seed(seed=5)
    test_alphas = list(np.linspace(-np.pi, -.3, 10)) + list(np.linspace(-.29, .29, 10)) + list(
        np.linspace(.3, np.pi, 10))
    test_alphas = np.array(test_alphas)
    I = test_planet.light_curve(test_alphas)
    errs = 0 * I + 1e-7
    noise_vals = np.random.normal(size=len(test_alphas))
    data_vals = errs * noise_vals + I
    data = np.array([test_alphas, data_vals, errs])
    with open('old_data.json', 'w') as f:
        json.dump(data.tolist(), f, indent=4)
    return data

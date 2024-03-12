import functools
import json

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as tck
from iminuit.cost import LeastSquares
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import AutoMinorLocator
import tqdm
import scipy.optimize as op
import time
from multiprocessing import Pool, freeze_support
import matplotlib.patches as mpatches
import pickle
import exoring_functions
import exoring_objects
from iminuit import Minuit

with open('constants.json') as json_file:
    constants = json.load(json_file)

with open('scattering_laws.pkl', 'rb') as f:
    scattering_laws = pickle.load(f)


# 2 not included because it does fit nicely on a graph
class FittingPlanet(exoring_objects.Planet):
    def __init__(self, sc_law, star, parameters):
        radius = parameters['radius']
        exoring_objects.Planet.__init__(self, sc_law, radius, star)


class FittingRingedPlanet(exoring_objects.RingedPlanet, FittingPlanet):
    def __init__(self, planet_sc_law, ring_sc_law, star, parameters):
        FittingPlanet.__init__(self, planet_sc_law, star, parameters)
        try:
            n_x, n_y, n_z = parameters['n_x'], parameters['n_y'], parameters['n_z']
            try:
                theta, phi = parameters['theta'], parameters['phi']
                raise Exception('Both n_x, n_y, n_z as well as theta and phi provided, please use 1 convention')
            except KeyError:
                pass
        except KeyError:
            try:
                theta, phi = parameters['theta'], parameters['phi']
                n_x = np.sin(theta) * np.cos(phi)
                n_y = np.sin(theta) * np.sin(phi)
                n_z = np.cos(theta)
            except KeyError:
                raise Exception('No ring orientation provided')
        disk_gap, ring_width = parameters['disk_gap'], parameters['ring_width']
        if n_x < 0:
            print('n_x was inputted to be <0, possibly the minimiser not respecting bounds')
        exoring_objects.RingedPlanet.__init__(self, self.sc_law, self.radius, ring_sc_law, self.radius + disk_gap,
                                              self.radius + disk_gap + ring_width, [n_x, n_y, n_z], star)


def gaussian(x, mu, sigma):
    with np.errstate(under='ignore'):
        a = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        if True in np.isinf(a):
            raise Exception('WasInf')
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


class DataObject:
    def __init__(self, data, star):
        self.best_result_ring = None
        self.best_result_planet = None
        self.data = data
        self.star = star

    def log_likelihood_planet(self, sc_law, *params):  # used for fitting
        # Calculates log likelihood for specific planet params and scattering law
        alpha = self.data[0]
        I = self.data[1]
        I_errs = self.data[2]
        parameters = {'radius': params[0]}
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

    def log_likelihood_ring(self, planet_sc_law, ring_sc_law,
                            *params):  # used for fitting
        # Calculates log likelihood for specific ringed planet params and scattering laws
        alpha = self.data[0]
        I = self.data[1]
        I_errs = self.data[2]
        parameters = {'radius': params[0], 'disk_gap': params[1], 'ring_width': params[2], 'n_x': params[3],
                      'n_y': params[4], 'n_z': params[5]}
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
        cost_function = LeastSquares(self.data[0], self.data[1], self.data[2], lambda alphas, radius: self.run_ringless_model(alphas, radius, planet_sc_law=planet_sc_law))
        minimiser = Minuit(cost_function, **init_guess)
        minimiser.migrad()
        minimiser.hesse()
        return minimiser

    def fitwrap_planet(self, args):
        # pool.map only takes a function with a single argument, so all arguments are wrapped here and parsed onto
        # the fitting function
        planet_sc_law = args[0]
        init_guess_planet = args[1]
        return self.fit_data_planet(planet_sc_law, init_guess_planet)

    def fit_data_ring(self, planet_sc_law, ring_sc_law, init_guess: dict, bounds: dict = None):
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
        cost_function = LeastSquares(self.data[0], self.data[1], self.data[2], lambda alphas, radius, disk_gap, ring_width, theta, phi: self.run_ringed_model(alphas, radius, disk_gap, ring_width, theta, phi, planet_sc_law=planet_sc_law, ring_sc_law=ring_sc_law))
        minimiser = Minuit(cost_function, **init_guess)
        if bounds:
            for key, bound in bounds.items():
                minimiser.limits[key] = bound
        minimiser.migrad()
        minimiser.hesse()
        return minimiser

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
                        results = list(
                            tqdm.tqdm(pool.imap(self.fitwrap_ring, args), total=len(args), desc='Running Models'))
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
                    results = list(
                        tqdm.tqdm(pool.imap(self.fitwrap_planet, args), total=len(args), desc='Running Models'))
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

    def run_ringless_model(self, alphas, radius, planet_sc_law):
        planet_sc_law = scattering_laws[f'{planet_sc_law}']
        model_planet = exoring_objects.Planet(planet_sc_law, radius, self.star)
        resulting_lightcurve = model_planet.light_curve(alphas)
        return resulting_lightcurve

    def run_ringed_model(self, alphas, radius, disk_gap, ring_width, theta, phi, planet_sc_law, ring_sc_law):
        planet_sc_law = scattering_laws[f'{planet_sc_law}']
        ring_sc_law = scattering_laws[f'{ring_sc_law}']
        n_x = np.sin(theta) * np.cos(phi)
        n_y = np.sin(theta) * np.sin(phi)
        n_z = np.cos(theta)
        model_ringed_planet = exoring_objects.RingedPlanet(planet_sc_law, radius, ring_sc_law, radius+disk_gap, radius+disk_gap+ring_width, [n_x, n_y, n_z], self.star)
        resulting_lightcurve = model_ringed_planet.light_curve(alphas)
        return resulting_lightcurve

    def display_ringless_model(self, planet_sc_law, model_parameters):
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

    def display_ringed_model(self, planet_sc_law, ring_sc_law, model_parameters, largest_diff=True, show_diff=True):
        planet_sc_law = scattering_laws[f'{planet_sc_law}']
        ring_sc_law = scattering_laws[f'{ring_sc_law}']
        if show_diff and not largest_diff:
            raise Exception('Largest_diff set to False but show_diff set to True')
        plt.style.use('poster.mplstyle')
        fig, ax = exoring_functions.generate_plot_style()
        plt.style.use('poster.mplstyle')
        model_parameters['n_x'], model_parameters['n_y'], model_parameters['n_z'] = model_parameters['ring_normal'][0], \
            model_parameters['ring_normal'][1], model_parameters['ring_normal'][2]
        I = self.data[1]
        I_errs = self.data[2]
        if model_parameters['n_x'] < 0:
            print('n_x was inputted to be <0')
        model_ringed_planet = FittingRingedPlanet(planet_sc_law, ring_sc_law, self.star, model_parameters)
        neg_alphas = np.linspace(-np.pi, 0, 5000)
        pos_alphas = np.linspace(0, np.pi, 5000)[1:]  # slicing to prevent repeat of alpha 0
        alphas = np.concatenate((neg_alphas, pos_alphas))
        resulting_lightcurve = model_ringed_planet.light_curve(alphas)
        resulting_lightcurve /= self.star.luminosity
        ax.errorbar(self.data[0] / np.pi, I / self.star.luminosity, I_errs / self.star.luminosity, fmt='.')
        ax.plot(alphas / np.pi, resulting_lightcurve)
        if largest_diff:
            neg_lightcurve = resulting_lightcurve[0:len(neg_alphas) - 1]  # -1 to ignore 0 alpha
            pos_lightcurve = resulting_lightcurve[len(neg_alphas):len(alphas)]
            neg_alphas, neg_lightcurve = neg_alphas[::-1], neg_lightcurve[
                                                           ::-1]  # Flip it, so we can work from closest to 0 alpha
            largest_diff = 0
            for neg_alpha, pos_alpha, neg_element, pos_element in zip(neg_alphas[1:], pos_alphas, neg_lightcurve,
                                                                      pos_lightcurve):  # slicing to prevent repeat alpha 0
                diff = abs(neg_element - pos_element)
                if diff > largest_diff:
                    largest_diff = diff
                    diff_neg_alpha = neg_alpha
                    diff_pos_alpha = pos_alpha
                    diff_neg_element = neg_element
                    diff_pos_element = pos_element
            if largest_diff != 0:
                botx, topx = ax.get_xlim()
                boty, topy = ax.get_ylim()
                if show_diff:
                    ax.vlines(np.array([diff_neg_alpha, diff_pos_alpha]) / np.pi, [boty, boty],
                              [diff_neg_element, diff_pos_element], linestyles='--')
                    ax.hlines([diff_neg_element, diff_pos_element], [botx, botx],
                              np.array([diff_neg_alpha, diff_pos_alpha]) / np.pi, linestyles='--')
                    arrow = mpatches.FancyArrowPatch((-1, diff_neg_element), (-1, diff_pos_element),
                                                     mutation_scale=20, arrowstyle='<|-|>', connectionstyle='Arc')
                    ax.add_patch(arrow)
                    ax.text(-1 + 0.2, (diff_pos_element + diff_neg_element) / 2, '{:.3g}'.format(largest_diff),
                            horizontalalignment='center', verticalalignment='center')
                ax.set_xlim(botx, topx)
                ax.set_ylim(boty, topy)
                print('Largest diff', largest_diff)
            else:
                print('No largest diff')

        plt.savefig('images/Ringed_Model', dpi=600)

    def display_many_ringless_models(self, planet_sc_law, multiple_model_parameters, sharex=True, sharey=False):
        length = len(multiple_model_parameters)
        nrows = np.sqrt(length)
        if nrows.is_integer():
            ncols = nrows
            plt.style.use('the_usual.mplstyle')
            fig, axs = plt.subplots(int(nrows), int(ncols), sharex=sharex, sharey=sharey)
        else:
            if length not in constants['primes_to_100']:
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
            elif length in constants['primes_to_100']:
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

    def display_many_ringed_models(self, planet_sc_law, ring_sc_law, multiple_model_parameters, static_parm,
                                   changing_parms, sharex=True, sharey=False):
        length = len(multiple_model_parameters)
        nrows = np.sqrt(length)
        row_parms = list()
        col_parms = list()
        if nrows.is_integer():
            ncols = nrows
            plt.style.use('the_usual.mplstyle')
            fig, axs = plt.subplots(int(nrows), int(ncols), sharex=sharex, sharey=sharey)
        else:
            if length not in constants['primes_to_100']:
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
            elif length in constants['primes_to_100']:
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
        # fig.tight_layout()
        fig.suptitle(
            f'Planet: {planet_sc_law.__name__} | Ring: {ring_sc_law.__name__} | {static_parm[0]}: {static_parm[1]}',
            y=0.995)
        for model_parameters, ax in zip(multiple_model_parameters, axs.flat):
            I = self.data[1]
            I_errs = self.data[2]
            model_parameters['n_x'], model_parameters['n_y'], model_parameters['n_z'] = model_parameters['ring_normal'][
                0], model_parameters['ring_normal'][1], model_parameters['ring_normal'][2]
            model_ringless_planet = FittingPlanet(planet_sc_law, self.star, model_parameters)
            model_ringed_planet = FittingRingedPlanet(planet_sc_law, ring_sc_law, self.star, model_parameters)
            alphas = np.linspace(-np.pi, np.pi, 10000)
            resulting_lightcurve = model_ringed_planet.light_curve(alphas)
            planet_lightcurve = model_ringless_planet.light_curve(alphas)
            # ax.errorbar(self.data[0] / np.pi, I, I_errs, fmt='.')
            ax.plot(alphas / np.pi, planet_lightcurve, 'orange')
            ax.plot(alphas / np.pi, resulting_lightcurve)
            # ax.set_title(
            #    f'R:{round(model_parameters['radius'],3)} G:{round(model_parameters['disk_gap'],3)} W:{round(model_parameters['ring_width'],3)}',
            #    fontsize=8, pad=2)
            if ax in axs[0]:  # If in the first row
                col_parms.append(model_parameters[changing_parms[0]])
                if changing_parms[0] == 'radius':
                    ax.set_title(f'R: {model_parameters[changing_parms[0]]}', color='b')
                elif changing_parms[0] == 'ring_width':
                    if changing_parms[1] == 'radius':
                        ax.set_title(f'W: {model_parameters[changing_parms[0]]}', color='b')
                elif changing_parms[0] == 'disk_gap':
                    ax.set_title(f'G: {model_parameters[changing_parms[0]]}', color='b')
                else:
                    ax.set_title(f'{changing_parms[0]}: {model_parameters[changing_parms[0]]}', color='b')
            for row in range(len(axs)):
                if ax == axs[row][int(ncols) - 1]:  # If in the last column
                    row_parms.append(model_parameters[changing_parms[1]])
                    ax2 = ax.twinx()
                    if changing_parms[1] == 'ring_width':
                        ax2.set_ylabel(f'W: {round(model_parameters[changing_parms[1]], 3)}', color='b', loc='center')
                    elif changing_parms[1] == 'radius':
                        ax2.set_ylabel(f'R: {round(model_parameters[changing_parms[1]], 3)}', color='b', loc='center')
                    elif changing_parms[1] == 'disk_gap':
                        ax2.set_ylabel(f'G: {round(model_parameters[changing_parms[1]], 3)}', color='b', loc='center')
                    else:
                        ax2.set_ylabel(f'{changing_parms[1]}: {model_parameters[changing_parms[1]]}', color='b')
                    # ax.set_title

        if int(nrows) == 1:
            for ax in axs:
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # force scientific notation
                ax.xaxis.set_major_formatter(FuncFormatter(exoring_functions.format_fraction_with_pi))
                ax.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
                ax.set_xlabel(r'Phase angle $\alpha$')
                for row in range(int(nrows)):
                    if ax == axs[0]:  # If in the first column
                        ax.set_ylabel(r'Intensity ($L_{\odot}$)')
        else:
            for ax in axs.flat:
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # force scientific notation
                ax.xaxis.set_major_formatter(FuncFormatter(exoring_functions.format_fraction_with_pi))
                ax.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
                if ax in axs[int(nrows) - 1]:  # If in the last row
                    ax.set_xlabel(r'Phase angle $\alpha$')
                for row in range(int(nrows)):
                    if ax == axs[row][0]:  # If in the first column
                        ax.set_ylabel(r'Intensity ($L_{\odot}$)')

        plt.savefig('images/Multiples', dpi=1000)

    def display_disperse_models(self, planet, planet_sc_law, ring_sc_law, changing_parms, model_parameters, sharex=True,
                                sharey=False):
        model_parameters['n_x'], model_parameters['n_y'], model_parameters['n_z'] = model_parameters['ring_normal'][
            0], model_parameters['ring_normal'][1], model_parameters['ring_normal'][2]
        I = self.data[1]
        I_errs = self.data[2]
        parm_values = list()
        planet_mass = (4 / 3) * np.pi * planet.radius ** 3  # Assuming density of  1 for planet
        hill_rad = self.star.distance * np.cbrt(planet_mass / (3 * (planet_mass + self.star.mass)))
        for parm in changing_parms:
            if parm == 'radius':
                parm_values.append((0.09, 0.83, 1))
            elif parm == 'disk_gap':
                parm_values.append(
                    (constants['SATURN-LIKE_RING_IN_R_JUP'] * planet.radius, 10 * planet.radius, (1 / 3) * hill_rad))
            elif parm == 'ring_width':
                parm_values.append(
                    (constants['SATURN-LIKE_RING_IN_R_JUP'] * planet.radius, 10 * planet.radius, (1 / 3) * hill_rad))
            else:
                raise NotImplementedError('Variable not implemented')
        meshes = np.meshgrid(*parm_values)
        flipped_axes_meshes = list()
        for mesh in meshes:
            flipped_axes_meshes.append(np.swapaxes(mesh, 0, 1))
        parm1mesh, parm2mesh = flipped_axes_meshes
        plt.style.use('the_usual.mplstyle')
        fig, axs = plt.subplots(3, 3, sharex=True)
        fig.suptitle(
            f'Planet: {planet_sc_law.__name__} | Ring: {ring_sc_law.__name__}',
            y=0.995)
        for row in range(len(parm1mesh)):
            for col in range(len(parm1mesh[0])):
                parm1, parm2 = parm1mesh[row][col], parm2mesh[row][col]
                ax = axs[row][col]
                parm1name, parm2name = changing_parms[0], changing_parms[1]
                altered_model = model_parameters.copy()
                altered_model[parm1name], altered_model[parm2name] = parm1, parm2
                model_ringless_planet = FittingPlanet(planet_sc_law, self.star, altered_model)
                model_ringed_planet = FittingRingedPlanet(planet_sc_law, ring_sc_law, self.star, altered_model)
                alphas = np.linspace(-np.pi, np.pi, 10000)
                ringed_lightcurve = model_ringed_planet.light_curve(alphas)
                planet_lightcurve = model_ringless_planet.light_curve(alphas)
                ax.plot(alphas / np.pi, planet_lightcurve, 'orange')
                ax.plot(alphas / np.pi, ringed_lightcurve)
                # ax.errorbar(self.data[0] / np.pi, I, I_errs, fmt='.')
                if row == 0:  # first row
                    if parm2name == 'disk_gap' or parm2name == 'ring_width':
                        if parm2 == constants['SATURN-LIKE_RING_IN_R_JUP'] * planet.radius:
                            if parm2name == 'disk_gap':
                                ax.set_title('G: Saturn', fontsize=14, pad=15)
                            elif parm2name == 'ring_width':
                                ax.set_title('W: Saturn', fontsize=14, pad=15)
                        elif parm2 == 10 * planet.radius:
                            if parm2name == 'disk_gap':
                                ax.set_title('G: 10x Planet', fontsize=14, pad=15)
                            elif parm2name == 'ring_width':
                                ax.set_title('W: 10x Planet', fontsize=14, pad=15)
                        elif parm2 == (1 / 3) * hill_rad:
                            if parm2name == 'disk_gap':
                                ax.set_title('G: 1/3 Hill', fontsize=14, pad=15)
                            elif parm2name == 'ring_width':
                                ax.set_title('W: 1/3 Hill', fontsize=14, pad=15)
                    elif parm2name == 'radius':
                        ax.set_title(f'R: {parm2}', fontsize=14, pad=15)
                if col == 2:  # last column
                    ax2 = ax.twinx()
                    if parm1name == 'disk_gap' or parm1name == 'ring_width':
                        if parm1 == constants['SATURN-LIKE_RING_IN_R_JUP'] * planet.radius:
                            if parm1name == 'disk_gap':
                                ax2.set_ylabel('G: Saturn', loc='center', fontsize=14, rotation=270, labelpad=20)
                            elif parm1name == 'ring_width':
                                ax2.set_ylabel('W: Saturn', loc='center', fontsize=14, rotation=270, labelpad=20)
                        elif parm1 == 10 * planet.radius:
                            if parm1name == 'disk_gap':
                                ax2.set_ylabel('G: 10x Planet', loc='center', fontsize=14, rotation=270, labelpad=20)
                            elif parm1name == 'ring_width':
                                ax2.set_ylabel('W: 10x Planet', loc='center', fontsize=14, rotation=270, labelpad=20)
                        elif parm1 == (1 / 3) * hill_rad:
                            if parm1name == 'disk_gap':
                                ax2.set_ylabel('G: 1/3 Hill', loc='center', fontsize=14, rotation=270, labelpad=20)
                            elif parm1name == 'ring_width':
                                ax2.set_ylabel('W: 1/3 Hill', loc='center', fontsize=14, rotation=270, labelpad=20)
                    elif parm1name == 'radius':
                        ax2.set_ylabel(f'R: {parm1}', loc='center', fontsize=14, rotation=270, labelpad=20)
                    ax2.tick_params(
                        axis='y',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        right=False,
                        left=False,
                        labelright=False)
        for ax in axs.flat:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # force scientific notation
            ax.yaxis.get_offset_text().set_fontsize(12)
            ax.xaxis.set_major_formatter(FuncFormatter(exoring_functions.format_fraction_with_pi))
            ax.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
            ax.tick_params(labelsize=13)
            if ax in axs[2]:  # If in the last row
                ax.set_xlabel(r'Phase angle $\alpha$', fontsize=12)
            for row in range(3):
                if ax == axs[row][0]:  # If in the first column
                    ax.set_ylabel(r'Intensity ($L_{\odot}$)', fontsize=12)
        fig.align_ylabels()
        # plt.tight_layout()
        plt.savefig('images/Disperse', dpi=1000)

    def create_various_model_parameters(self, parameters_to_vary):
        # Changing albedos not implemented
        all_dicts = list()
        all_params = list()
        order_dictionary = dict()
        default = {'radius': 1,
                   'disk_gap': 1, 'ring_width': 1,
                   'ring_normal': np.array([1., 1., 0])}
        for order, key in enumerate(parameters_to_vary):
            param_list = list()
            param_list.append(parameters_to_vary[key])
            all_params.append(param_list)
            order_dictionary[order] = key
        grid = np.meshgrid(*all_params)
        positions = np.vstack(list(map(np.ravel, grid)))
        for iteration in range(len(positions[0])):
            new_dict = default.copy()
            for order in range(len(positions)):
                key = order_dictionary[order]
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

    def wrapped_likelihood_ringless_model(self, args):
        likelihood = self.likelihood_ringless_model(*args[1:])
        indices = args[0]
        return indices, likelihood

    def likelihood_ringed_model(self, planet_sc_law, ring_sc_law, model_parameters):  # Used for manual models
        # Calculates log likelihood for specific ringed planet params and scattering laws
        planet_sc_law = scattering_laws[f'{planet_sc_law}']
        ring_sc_law = scattering_laws[f'{ring_sc_law}']
        alpha = self.data[0]
        I = self.data[1]
        I_errs = self.data[2]
        try:
            model_parameters['n_x'], model_parameters['n_y'], model_parameters['n_z'] = model_parameters['ring_normal'][
                0], \
                model_parameters['ring_normal'][1], model_parameters['ring_normal'][2]
            try:
                theta, phi = model_parameters['theta'], model_parameters['phi']
                raise Exception('Both normal as well as theta and phi provided, please use 1 convention')
            except KeyError:
                pass
        except KeyError:
            try:
                model_parameters['theta'], model_parameters['phi']
            except KeyError:
                raise Exception('No ring orientation provided')

        model_ringed_planet = FittingRingedPlanet(planet_sc_law, ring_sc_law, self.star, model_parameters)
        x = model_ringed_planet.light_curve(alpha)
        return np.prod(gaussian(x, I, I_errs))

    def wrapped_likelihood_ringed_model(self, args):
        likelihood = self.likelihood_ringed_model(*args[1:])
        indices = args[0]
        return indices, likelihood

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

    def produce_slice_plot(self, best_model, ranges, ringed, **kwargs):
        planet_sc_law = kwargs['planet_sc_law']
        if ringed:
            ring_sc_law = kwargs['ring_sc_law']
            best_ll = self.log_likelihood_ringed_model(planet_sc_law, ring_sc_law, best_model)
        else:
            best_ll = self.log_likelihood_ringless_model(planet_sc_law, best_model)
        mixed_keys = list()
        for key1 in ranges.keys():
            for key2 in ranges.keys():
                if key1 != key2 and (key2, key1) not in mixed_keys:
                    mixed_keys.append((key1, key2))
        for key1, key2 in mixed_keys:
            all_params = list()
            key1_value_range = ranges[key1]
            key2_value_range = ranges[key2]
            key1_values = np.linspace(key1_value_range[0], key1_value_range[1], 50)
            key2_values = np.linspace(key2_value_range[0], key2_value_range[1], 50)
            all_params.append(key1_values)
            all_params.append(key2_values)
            X, Y = np.meshgrid(*all_params)
            Z = np.zeros((len(X), len(X[0])))
            for row in range(len(X)):
                for column in range(len(X[0])):
                    altered_model = best_model.copy()
                    altered_model[key1] = X[row][column]
                    altered_model[key2] = Y[row][column]
                    if ringed:
                        log_likelihood = self.log_likelihood_ringed_model(planet_sc_law, ring_sc_law, altered_model)
                        Z[row][column] = log_likelihood
                    else:
                        log_likelihood = self.log_likelihood_ringless_model(planet_sc_law, altered_model)
                        Z[row][column] = log_likelihood

            plt.style.use('the_usual.mplstyle')
            fig, ax = plt.subplots()
            cp = ax.contourf(X, Y, Z, cmap='Blues')
            cbar = fig.colorbar(cp)  # Add a colorbar to a plot
            cbar.ax.tick_params(labelsize=12)
            ax.set_title(f'log likelihood {key2} against {key1}', fontsize=13)
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

    def produce_corner_plot(self, best_model, ranges, planet_sc_law, ring_sc_law=None, ringed=True, log=False,
                            multiprocessing=True, save_data = True):
        varname_to_dispname = {'theta':r'$\boldsymbol{\theta}$', 'phi': r'$\boldsymbol{\phi}$', 'radius': 'Radius', 'ring_width': 'Ring width', 'disk_gap': 'Ring gap'} # Used to display variable names in graphs
        '''
        if ringed:
            if not ring_sc_law:
                raise Exception('Ringed model requested but no ring_sc_law inputted')
            best_ll = self.likelihood_ringed_model(planet_sc_law, ring_sc_law, best_model)
        else:
            best_ll = self.likelihood_ringless_model(planet_sc_law, best_model)
        '''
        keys = ranges.keys()
        keyslist = list(keys)
        indices = list(range(len(keyslist)))
        keys_order = dict()
        all_params = list()
        for i, key in enumerate(keys):
            key_value_range = ranges[key]
            key_values = np.linspace(key_value_range[0], key_value_range[1], key_value_range[2])
            all_params.append(key_values)
            keys_order[key] = i
        mixed_indices = list()
        for index1 in indices:
            for index2 in indices:
                if index1 != index2 and (index2, index1) not in mixed_indices:
                    mixed_indices.append((index1, index2))
        meshes = np.meshgrid(*all_params)
        plotmeshes = dict()
        for index1, index2 in mixed_indices:
            plotmeshes[f'{index1}+{index2}'] = np.meshgrid(all_params[index1], all_params[index2])
        flipped_axes_meshes = list()
        for mesh in meshes:
            flipped_axes_meshes.append(np.swapaxes(mesh, 0, 1))
        meshes = flipped_axes_meshes
        likelihood = np.zeros(meshes[0].shape)
        # saved = dict() # Used to check whether x in np.trapz was correct
        indices_meshes = np.meshgrid(*[[i for i in range(likelihood.shape[j])] for j in range(len(likelihood.shape))])
        positions = np.vstack(list(map(np.ravel, indices_meshes)))
        positions = np.transpose(positions)
        start = time.time()
        if multiprocessing:
            args = list()
            for indexes in positions:
                altered_model = best_model.copy()
                for i, key in enumerate(keyslist):
                    altered_model[key] = meshes[i][*indexes]
                if ringed:
                    args.append((indexes, planet_sc_law, ring_sc_law, altered_model))
                else:
                    args.append((indexes, planet_sc_law, altered_model))
            with Pool(16) as pool:
                if ringed:
                    results = list(tqdm.tqdm(pool.imap(self.wrapped_likelihood_ringed_model, args), total=len(args),
                                             desc='Running Models', colour='green'))
                    for result in results:
                        likelihood[*result[0]] = result[1]
                    # saved[f'{indexes}'] = f'radius:{meshes[0][*indexes]} disk_gap:{meshes[1][*indexes]} ring_width:{meshes[2][*indexes]}' #Used to check whether x in np.trapz was correct

                else:
                    results = list(tqdm.tqdm(pool.imap(self.wrapped_likelihood_ringless_model, args), total=len(args),
                                             desc='Running Models', colour='green'))
                    for result in results:
                        likelihood[*result[0]] = result[1]
        else:
            for indexes in tqdm.tqdm(positions, desc='Running Models'):
                altered_model = best_model.copy()
                for i, key in enumerate(keyslist):
                    altered_model[key] = meshes[i][*indexes]
                if ringed:
                    likelihood_val = self.likelihood_ringed_model(planet_sc_law, ring_sc_law, altered_model)
                    likelihood[*indexes] = likelihood_val
                    # saved[f'{indexes}'] = f'radius:{meshes[0][*indexes]} disk_gap:{meshes[1][*indexes]} ring_width:{meshes[2][*indexes]}' #Used to check whether x in np.trapz was correct

                else:
                    likelihood_val = self.likelihood_ringless_model(planet_sc_law, altered_model)
                    likelihood[*indexes] = likelihood_val
        with open('likelihoods.json', 'w') as f:
            json.dump(likelihood.tolist(), f)
        end = time.time()
        print('Time taken for running of models', round(end - start, 2), 'seconds')
        '''
        previous_integral = likelihood
        for i in range(len(meshes)):
            if len(meshes)-i-1 == 0: # Last variable to integrate through
                total_integral = np.trapz(previous_integral, x=range(len(all_params[-(i + 1)])))
            else:
                integral_over_mesh = np.zeros(likelihood.shape[0:len(meshes)-i-1])
                indices_meshes = np.meshgrid(
                    *[[i for i in range(integral_over_mesh.shape[j])] for j in range(len(integral_over_mesh.shape))]) # list comprehension creates lists of possible index values.e.g. for ndarray:(2,4) --> [[0, 1], [0, 1, 2, 3]]
                if indices_meshes: # If not empty
                    positions = np.vstack(list(map(np.ravel, indices_meshes)))
                else:
                    positions = np.array([0])
                positions = np.transpose(positions)
                for indices in positions:
                    val = np.trapz(previous_integral[*indices], x=range(len(all_params[-(i+1)]))) # Not sure if x is right here, figure this out
                    integral_over_mesh[*indices] = val
                previous_integral = integral_over_mesh

        if total_integral != 0:
            likelihood = likelihood / total_integral
        '''
        plt.style.use('the_usual.mplstyle')
        if save_data == True:
            data = dict()
        fig, axs = plt.subplots(len(keyslist), len(keyslist), sharex='col')  # share the x-axis between columns
        for column, columnkey in zip(range(len(keyslist)), keyslist):
            for row, rowkey in zip(range(len(keyslist)), keyslist):
                if column > row:
                    fig.delaxes(axs[row][column])
                    continue
                elif row == column:  # diagonal, integrate over all variables but rowkey
                    rearranged_likelihood = np.swapaxes(likelihood, 0, keys_order[rowkey])
                    param_values = all_params.copy()
                    param_values[0], param_values[keys_order[rowkey]] = param_values[keys_order[rowkey]], param_values[
                        0]
                    previous_integral = rearranged_likelihood
                    for i in range(len(meshes)):
                        if len(meshes) - i - 1 == 0:  # Penultimate variable
                            integral_over_mesh = previous_integral
                            normalisation = np.trapz(previous_integral, x=param_values[0])
                            if normalisation != 0:
                                integral_over_mesh /= normalisation
                            break
                        else:
                            integral_over_mesh = np.zeros(rearranged_likelihood.shape[0:len(meshes) - i - 1])
                            indices_meshes = np.meshgrid(
                                *[[i for i in range(integral_over_mesh.shape[j])] for j in range(
                                    len(integral_over_mesh.shape))])  # list comprehension creates lists of possible index values.e.g. for ndarray:(2,4) --> [[0, 1], [0, 1, 2, 3]]
                            positions = np.vstack(list(map(np.ravel, indices_meshes)))
                            positions = np.transpose(positions)
                            for indices in positions:
                                val = np.trapz(previous_integral[*indices],
                                               x=param_values[-(i + 1)])  # Not sure if x is right here, figure this out
                                integral_over_mesh[*indices] = val
                            previous_integral = integral_over_mesh
                    if columnkey == 'theta' or columnkey == 'phi':
                        axs[row][column].plot(param_values[0]/np.pi, integral_over_mesh, color='#084d96') # Formatting angles in terms of pi for visual clarity
                        axs[row][column].set_xlim(param_values[0][0]/np.pi, param_values[0][-1]/np.pi)
                    else:
                        axs[row][column].plot(param_values[0], integral_over_mesh, color='#084d96')
                        axs[row][column].set_xlim(param_values[0][0], param_values[0][-1])
                    axs[row][column].tick_params(
                        axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=True,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=True, labelsize=12)
                    axs[row][column].set_ylim(0)
                    if save_data == True:
                        data[f'{rowkey}'] = (param_values[0], integral_over_mesh)
                else:  # contour, integrate over all variables but rowkey and columnkey
                    first_rearranged_likelihood = np.swapaxes(likelihood, 0, keys_order[columnkey])
                    rearranged_likelihood = np.swapaxes(first_rearranged_likelihood, 1, keys_order[rowkey])
                    param_values = all_params.copy()
                    if keys_order[columnkey] == 1:
                        if keys_order[rowkey] == 0:
                            param_values[0], param_values[1] = param_values[1], param_values[0]
                        else:
                            param_values[0], param_values[1], param_values[keys_order[rowkey]] = param_values[1], \
                            param_values[keys_order[rowkey]], param_values[0]
                    elif keys_order[columnkey] == 0:
                        if keys_order[rowkey] == 1:
                            pass
                        else:
                            param_values[1], param_values[keys_order[rowkey]] = param_values[keys_order[rowkey]], \
                            param_values[1]
                    elif keys_order[rowkey] == 0:
                        if keys_order[columnkey] != 1:
                            param_values[1], param_values[0], param_values[keys_order[columnkey]] = param_values[0], \
                            param_values[keys_order[columnkey]], param_values[1]
                        else:
                            raise Exception('This should be impossible, should have been treated earlier')
                    elif keys_order[rowkey] == 1:
                        if keys_order[columnkey] != 0:
                            param_values[0], param_values[keys_order[columnkey]] = param_values[keys_order[columnkey]], \
                            param_values[0]
                        else:
                            raise Exception('This should be impossible, should have been treated earlier')
                    else:
                        param_values[0], param_values[1], param_values[keys_order[columnkey]], param_values[
                            keys_order[rowkey]] = \
                            param_values[keys_order[columnkey]], param_values[keys_order[rowkey]], param_values[0], \
                            param_values[1]
                    previous_integral = rearranged_likelihood
                    for i in range(len(meshes)):
                        if len(meshes) - i - 1 == 1:  # Penultimate variable
                            integral_over_mesh = previous_integral
                            next_integral = np.zeros(len(param_values[0]))
                            for i in range(len(param_values[0])):
                                next_integral[i] = np.trapz(previous_integral[i], x=param_values[1])
                            normalisation = np.trapz(next_integral, x=param_values[0])
                            if normalisation != 0:
                                integral_over_mesh /= normalisation
                            break
                        else:
                            integral_over_mesh = np.zeros(rearranged_likelihood.shape[0:len(meshes) - i - 1])
                            indices_meshes = np.meshgrid(
                                *[[i for i in range(integral_over_mesh.shape[j])] for j in range(
                                    len(integral_over_mesh.shape))])  # list comprehension creates lists of possible index values.e.g. for ndarray:(2,4) --> [[0, 1], [0, 1, 2, 3]]
                            positions = np.vstack(list(map(np.ravel, indices_meshes)))
                            positions = np.transpose(positions)
                            for indices in positions:
                                val = np.trapz(previous_integral[*indices],
                                               x=param_values[-(i + 1)])  # Not sure if x is right here, figure this out
                                integral_over_mesh[*indices] = val
                            previous_integral = integral_over_mesh
                    contour_array = integral_over_mesh
                    number_of_steps = 1000
                    if log == True:
                        with np.errstate(divide='ignore'):
                            contour_array = np.log(contour_array)
                        if contour_array[contour_array != -np.inf].min() < 0:
                            contour_array[np.isinf(contour_array)] = 1.1 * contour_array[contour_array != -np.inf].min()
                        elif contour_array[contour_array != -np.inf].min() > 0:
                            contour_array[np.isinf(contour_array)] = contour_array[contour_array != -np.inf].min() / 10
                        else:
                            raise NotImplementedError
                        number_of_steps = 10000

                    step = (contour_array.max() - contour_array.min()) / number_of_steps
                    if not log:
                        if contour_array.max() == 0:
                            levels = np.linspace(0, 1, 10)
                        else:
                            levels = np.arange(start=contour_array.min(), stop=contour_array.max(), step=step)
                    else:
                        levels = np.arange(start=contour_array.min(), stop=contour_array.max(), step=step)
                    plt.style.use('the_usual.mplstyle')
                    if columnkey == 'theta' or columnkey == 'phi':
                        axs[row][column].contourf(
                            np.swapaxes(plotmeshes[f'{keys_order[columnkey]}+{keys_order[rowkey]}'][0], 0, 1)/np.pi, # Formatting angles in terms of pi
                            np.swapaxes(plotmeshes[f'{keys_order[columnkey]}+{keys_order[rowkey]}'][1], 0, 1)/np.pi,
                            contour_array, levels,
                            cmap='Blues')
                    else:
                        axs[row][column].contourf(
                            np.swapaxes(plotmeshes[f'{keys_order[columnkey]}+{keys_order[rowkey]}'][0], 0, 1),
                            np.swapaxes(plotmeshes[f'{keys_order[columnkey]}+{keys_order[rowkey]}'][1], 0, 1),
                            contour_array, levels,
                            cmap='Blues')

                    if save_data == True:
                        data[f'{rowkey} and {columnkey}'] = (np.swapaxes(plotmeshes[f'{keys_order[columnkey]}+{keys_order[rowkey]}'][0], 0, 1), np.swapaxes(plotmeshes[f'{keys_order[columnkey]}+{keys_order[rowkey]}'][1], 0, 1), contour_array)
                    # cbar = fig.colorbar(cp)  # Add a colorbar to a plot
                    # cbar.ax.tick_params(labelsize=12)
                    # axs[row][column].yaxis.set_minor_locator(AutoMinorLocator(2))
                if axs[row][column] in axs[len(keyslist) - 1]:  # If in the last row
                    axs[row][column].set_xlabel(varname_to_dispname[f'{columnkey}'], loc='center')
                    axs[row][column].tick_params(direction='in', left=False, bottom=True, top=False, right=False,
                                                 which='both', labelsize=12)
                    axs[row][column].minorticks_on()
                    if columnkey == 'theta' or columnkey == 'phi':
                        axs[row][column].xaxis.set_major_formatter(FuncFormatter(exoring_functions.format_fraction_with_pi))
                        axs[row][column].xaxis.set_major_locator(tck.MultipleLocator(base=1/4))
                    if columnkey == 'phi':
                        axs[row][column].xaxis.set_minor_locator(AutoMinorLocator(2))
                else:
                    axs[row][column].tick_params(
                        axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=True,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False, labelsize=12)
                    axs[row][column].minorticks_on()
                if column == 0 and row != 0:
                    axs[row][column].set_ylabel(varname_to_dispname[rowkey], loc='center')
                    axs[row][column].tick_params(axis='y', direction='in', left=True, bottom=False, top=False,
                                                 right=False,
                                                 which='both', labelsize=12, colors='black')
                else:
                    axs[row][column].tick_params(
                        axis='y',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        left=False,  # ticks along the bottom edge are off
                        right=False,  # ticks along the top edge are off
                        labelleft=False, labelsize=12)
        fig.align_xlabels()
        fig.align_ylabels()
        #fig.tight_layout()
        plt.savefig('images/corner_plot', dpi=600)
        if save_data == True:
            with open('corner_data.pkl', 'wb') as f:
                pickle.dump(data, f)


def generate_data(test_planet):
    np.random.seed(seed=5)
    test_alphas = list(np.linspace(-np.pi, -.3, 10)) + list(np.linspace(-.29, .29, 10)) + list(
        np.linspace(.3, np.pi, 10))
    test_alphas = np.array(test_alphas)
    I = test_planet.light_curve(test_alphas)
    errs = 0 * I + 1e-7 * test_planet.star.luminosity
    noise_vals = np.random.normal(size=len(test_alphas))
    data_vals = errs * noise_vals + I
    data = np.array([test_alphas, data_vals, errs])
    return data

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:37:10 2023

@author: victo
"""
import time

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
'''
# %%

# basic curve_fit stuff

star = exoring_objects.Star(1, SUN_TO_JUP, 0.1 * AU_TO_JUP, 1)


def fit_func_ring(alphas, r_planet, planet_albedo, r_inner, ring_width, ring_angle, ring_albedo, X, m):
    sc_law_planet = scattering.Lambert(planet_albedo)
    model_planet = exoring_objects.Planet(sc_law_planet, r_planet, star)
    ring_normal = [np.cos(ring_angle), np.sin(ring_angle), 0]
    sc_law_ring = scattering.Mie(ring_albedo, X, m)
    model_ring = exoring_objects.Ring(sc_law_ring, r_inner, r_inner + ring_width, ring_normal, star)
    return model_planet.light_curve(alphas) + model_ring.light_curve(alphas)


def fit_func_planet(alphas, r, albedo):
    sc_law = scattering.Lambert(albedo)
    model_planet = exoring_objects.Planet(sc_law, r, star)
    return model_planet.light_curve(alphas)


test_planet = exoring_objects.Planet(scattering.Lambert(1), 1, star)
test_ring = exoring_objects.Ring(scattering.Rayleigh(1), 2, 3, [1, 0.8, 0.1], star)

alphas = np.linspace(-np.pi, np.pi, 10000)

data = test_planet.light_curve(alphas) + test_ring.light_curve(alphas)

ring_vals, ring_cov = op.curve_fit(fit_func_ring, alphas, data, p0=[1, 0.8, 1, 1.1, 0.2, 1, 0.01, 1.5], bounds=(
    [0, 0, 0, 0, 0, 0, 0, 1.], [np.inf, np.inf, np.inf, np.inf, 2 * np.pi, np.inf, np.inf, np.inf]))
planet_vals, planet_cov = op.curve_fit(fit_func_planet, alphas, data, p0=[2, 0.8])

plt.style.use('the_usual')

plt.plot(alphas, data, label='Data')
plt.plot(alphas, fit_func_ring(alphas, *ring_vals), label='Fitted planet with ring')
plt.plot(alphas, fit_func_planet(alphas, *planet_vals), label='Fitted planet')

plt.legend()

# %%

star = exoring_objects.Star(1, SUN_TO_JUP, 0.1 * AU_TO_JUP, 1)
'''

def gaussian(x, mu, sigma):
    return (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-0.5 * ((x - mu) / (sigma)) ** 2)


class Fitting():
    def __init__(self, data, star, planet_scattering_class=scattering.HG, ring_scattering_class=scattering.Mie):
        self.data = data
        self.star = star
        self.planet_scattering_class = planet_scattering_class
        self.ring_scattering_class = ring_scattering_class

    def nll_ring(self, r_planet, r_inner, ring_width, ring_angle, **kwargs):
        alpha = self.data[0]
        I = self.data[1]
        I_errs = self.data[2]
        sc_planet = self.planet_scattering_class(**kwargs)
        # sc_planet = scattering.HG(g, albedo_planet)
        n_ring = [np.cos(ring_angle), np.sin(ring_angle), 0]
        sc_ring = self.ring_scattering_class(**kwargs)
        # sc_ring = scattering.Mie(albedo_ring, X, m)
        model_planet = exoring_objects.Planet(sc_planet, r_planet, self.star)
        model_ring = exoring_objects.Ring(sc_ring, r_inner, r_inner + ring_width, n_ring, self.star)
        x = model_planet.light_curve(alpha) + model_ring.light_curve(alpha)
        return -np.sum(np.log(gaussian(x, I, I_errs)))

    def quasi_newton_minimisation(self, func, stepsize, sigma, *args, **kwargs):
        start = time.time()
        init_vals = np.array(args)
        kwarg_names = list(kwargs.keys())
        kwargs_ordering = {number + len(args): kwarg_names[number] for number in range(len(kwargs))}
        for kwarg in kwarg_names:
            init_vals = np.append(init_vals, kwargs[kwarg])
        update_matrix = np.identity(len(init_vals))
        x_0_grad = grad(func, init_vals, 1e-10, kwargs_ordering)
        x_1 = init_vals - stepsize * x_0_grad
        delta = x_1 - init_vals
        x_1_grad = grad(func, x_1, 1e-10, kwargs_ordering)
        gamma = x_1_grad - x_0_grad
        x_0 = x_1
        while True:
            update_matrix = update_matrix + (np.outer(delta, delta)) / (
                np.dot(gamma, delta)) - (
                                np.matmul(update_matrix,
                                          np.matmul(np.outer(gamma, gamma), update_matrix))) / (
                                np.dot(gamma, np.matmul(update_matrix, gamma)))
            x_1 = x_0 - stepsize * np.matmul(update_matrix, x_1_grad)
            if (np.linalg.norm(x_1) - np.linalg.norm(x_0)) / np.linalg.norm(x_0) <= sigma or time.time() - start >= 100:
                return x_1
            else:
                x_0_grad = x_1_grad
                x_1_grad = grad(func, x_1, 1e-10, kwargs_ordering)
                delta = x_1 - x_0
                gamma = x_1_grad - x_0_grad
                x_0 = x_1


def first_order_partial_derivative(func, position, variable_num, stepsize, kwargs_ordering):
    position = np.array(position)
    args = position[:max(kwargs_ordering.keys())]
    if variable_num < max(kwargs_ordering.keys()):  # derivative basic arguments
        step = np.zeros((len(args),))
        step[variable_num] += stepsize
        forward_args = args + step
        backward_args = args - step
        kwargs = {kwarg_name: position[i] for i, kwarg_name in kwargs_ordering.items()}
        forward_val = func(*forward_args, **kwargs)
        backward_val = func(*backward_args, **kwargs)
        derivative = (forward_val - backward_val) / (2 * stepsize)
    else:
        forward_kwarg = position[variable_num] + stepsize
        backward_kwarg = position[variable_num] - stepsize
        forward_kwargs_dict = {kwargs_ordering[index + len(args)]: (position[index + len(args)] if
                                                                    index + len(
                                                                        args) != variable_num else forward_kwarg) for
                               index in range(len(kwargs_ordering))}
        backward_kwargs_dict = {kwargs_ordering[index + len(args)]: (position[index + len(args)] if
                                                                     index + len(
                                                                         args) != variable_num else backward_kwarg) for
                                index in range(len(kwargs_ordering))}
        forward_val = func(*args, **forward_kwargs_dict)
        backward_val = func(*args, **backward_kwargs_dict)
        derivative = (forward_val - backward_val) / (2 * stepsize)
    return derivative


def grad(func, position, stepsize, kwargs_ordering):
    gradient = [first_order_partial_derivative(func, position, variable_num, stepsize, kwargs_ordering) for
                variable_num in
                range(len(position))]

    return np.array(gradient)

'''
def minimize_logL(data, p0):
    m = op.minimize(lambda *p: nll_ring(data, *p), p0)
    return m.x
'''
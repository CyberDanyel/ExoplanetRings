import numpy as np
from fractions import Fraction

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as tck
import scipy.interpolate as spint


def integrate2d(func, bounds: list, sigma=0.01):
    """
    2D integration by basic Riemann sum

    Parameters
    ----------
    func : function object
        The function to be integrated, with two input variables
    bounds : iterable w. shape 2,2
        A tuple of floats specifying the bounds of the respective variables
    sigma : float
        Fractional error acceptable for end of integration
    Returns
    -------
    float
        The definite integral

    """
    old_total_integral = 0
    iteration = 1
    n = 1000
    width = int(np.sqrt(n))
    while True:
        xs = np.broadcast_to(np.linspace(bounds[0][0], bounds[0][1], width), (width, width))
        ys = np.broadcast_to(np.array([np.linspace(bounds[1][0], bounds[1][1], width)]).T, (width, width))
        area = (xs[0][1] - xs[0][0]) * (ys[1][0] - ys[0][0])
        arr = func(xs, ys)
        new_total_integral = np.sum(arr) * area
        if iteration == 1:
            n = n * 2
            width = int(np.sqrt(n))
            iteration += 1
            old_total_integral = new_total_integral
            pass
        else:
            if old_total_integral == 0 or abs((
                                                      new_total_integral - old_total_integral) / old_total_integral) < sigma or old_total_integral < 1e-10:
                return new_total_integral
            else:
                n = n * 2
                width = int(np.sqrt(n))
                iteration += 1
                old_total_integral = new_total_integral
                pass


class MonteCarloPlanetIntegration:
    def __init__(self, i):
        self.i = i
        xs = np.random.uniform(0, 1, self.i)
        self.thetas = np.arccos(1 - 2 * xs)
        # self.phis = np.random.uniform(- np.pi / 2, np.pi / 2,self.i)

    def integrate(self, alpha, func):
        if alpha >= 0:
            # phis = [phi for phi in self.phis if alpha - np.pi / 2 < phi < np.pi / 2]
            phis = np.random.uniform(alpha - np.pi / 2, np.pi / 2, self.i)
        else:
            # phis = [phi for phi in self.phis if -np.pi / 2 < phi < alpha + np.pi / 2]
            phis = np.random.uniform(-np.pi / 2, alpha + np.pi / 2, self.i)
        if alpha >= 0:
            qs = func(self.thetas, phis) * (2 * (np.pi - alpha)) / np.sin(self.thetas)
        else:
            qs = func(self.thetas, phis) * (2 * (np.pi + alpha)) / np.sin(self.thetas)
        q_average = sum(qs) / self.i
        # sigma_squared = (1 / (self.i - 1)) * sum((qs - q_average) ** 2)
        # sigma = np.sqrt(sigma_squared)
        # error = (1 / np.sqrt(self.i)) * sigma
        # fractional_error = error / q_average
        # print(f'q_average = {q_average} +-', f'{fractional_error*100}%')
        return q_average


def format_fraction_with_pi(x, pos):
    fract = Fraction(x).limit_denominator()
    if fract == 0:
        return "0"
    elif x == 1:
        return '$\\pi$'
    elif x == -1:
        return '$-\\pi$'
    else:
        if fract.numerator > 0:
            return f'$\\frac{{{fract.numerator}}}{{{fract.denominator}}}$' + '$\\pi$'
        else:
            return f'$-\\frac{{{abs(fract.numerator)}}}{{{fract.denominator}}}$' + '$\\pi$'


def monte_carlo_ring_integration(func, bounds_y, bounds_z, i):
    integration_area = abs(bounds_y[1] - bounds_y[0]) * abs(bounds_z[1] - bounds_z[0])
    ys = np.random.uniform(bounds_y[0], bounds_y[1], i)
    zs = np.random.uniform(bounds_z[0], bounds_z[1], i)
    func_values = func(ys, zs)
    added_func_values = sum(func_values)
    average_func_value = added_func_values / i
    integral = (integration_area / i) * added_func_values
    sampling_sigma_squared = (1 / (i - 1)) * sum((func_values - average_func_value) ** 2)
    integral_sigma = (integration_area / np.sqrt(i)) * np.sqrt(sampling_sigma_squared)
    return integral  # numerical errors may bring this down to 0


def circle_section_integral(radius, bounds: []):
    upper = radius ** 2 * np.arcsin(bounds[1] / radius) + bounds[1] * np.sqrt(radius ** 2 - bounds[1] ** 2)
    bottom = radius ** 2 * np.arcsin(bounds[0] / radius) + bounds[0] * np.sqrt(radius ** 2 - bounds[0] ** 2)
    integration_result = upper - bottom
    return integration_result


def overlap_area(r_circle, r_ellipse, mu, cos_phi, sin_phi, offset):
    def find_distance_from_ellipse_centre(a, b):
        with np.errstate(all='raise'):
            if mu == 0:
                return a ** 2 + b ** 2
            else:
                return (a * cos_phi - b * sin_phi) ** 2 + (1 / mu ** 2) * (a * sin_phi + b * cos_phi) ** 2

    angles = np.linspace(0, 2 * np.pi, 2000)
    xs = r_circle * np.cos(angles) + offset
    ys = r_circle * np.sin(angles)
    in_ellipse = (find_distance_from_ellipse_centre(xs, ys) < r_ellipse ** 2)
    x = xs[np.roll(in_ellipse, 1) != in_ellipse]
    y = ys[np.roll(in_ellipse, 1) != in_ellipse]

    if len(x) == 0:
        if np.abs(offset) < r_circle:
            return np.pi * mu * r_ellipse ** 2
        else:
            return 0.

    x_prime = (x * cos_phi + y * sin_phi)
    y_prime = (y * cos_phi - x * sin_phi) / mu
    # plt.scatter(x, y)
    # plt.scatter(x_prime, y_prime)

    circle_rot_angle = np.arctan((x[1] - x[0]) / (y[1] - y[0]))

    x_circle_prime = (x - offset) * np.cos(circle_rot_angle) - y * np.sin(circle_rot_angle)
    # y_circle_prime = (x-offset)*np.sin(circle_rot_angle) + y*np.cos(circle_rot_angle)
    # plt.scatter(x_circle_prime, y_circle_prime)

    circle_section_area = np.abs(
        circle_section_integral(r_circle, bounds=[x_circle_prime[0], - np.sign(offset) * r_circle]))

    ellipse_rot_angle = np.arctan((x_prime[1] - x_prime[0]) / (y_prime[1] - y_prime[0]))
    x_ellipse_prime = x_prime * np.cos(ellipse_rot_angle) - y_prime * np.sin(ellipse_rot_angle)
    y_ellipse_prime = x_prime * np.sin(ellipse_rot_angle) + y_prime * np.cos(ellipse_rot_angle)
    # plt.scatter(x_ellipse_prime, y_ellipse_prime)

    ellipse_section_area = mu * np.abs(
        circle_section_integral(r_ellipse,
                                bounds=[x_ellipse_prime[0], np.sign(offset) * r_ellipse]))
    return ellipse_section_area + circle_section_area


def generate_plot_style():
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(FuncFormatter(format_fraction_with_pi))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
    ax.set_xlabel(r'Phase angle $\alpha$')
    ax.set_ylabel(r'Intensity ($L_{\odot}$)')
    fig.tight_layout()
    return fig, ax

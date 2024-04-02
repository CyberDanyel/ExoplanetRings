import numpy as np
from fractions import Fraction

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as tck
import scipy.integrate as spi


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
        return '$\\text{{-}}\\pi$'
    else:
        if fract.numerator > 0:
            return f'$\\frac{{{fract.numerator}}}{{{fract.denominator}}}$' + '$\\pi$'
        else:
            return f'$-\\frac{{{abs(fract.numerator)}}}{{{fract.denominator}}}$' + '$\\pi$'

def format_fraction_with_pi_small(x, pos):
    fract = Fraction(x).limit_denominator()
    if fract == 0:
        return "0"
    elif x == 1:
        return '$\\pi$'
    elif x == -1:
        return '$-\\pi$'
    else:
        if fract.numerator > 0:
            if fract.numerator == 1:
                return f'$\\pi/{fract.denominator}$'
            else:
                return f'${fract.numerator}\\pi/{fract.denominator}$'
        else:
            return f'$-{abs(fract.numerator)}\\pi/{fract.denominator}$'

def format_fraction_with_r_jup(x, pos):
    fract = Fraction(x).limit_denominator()
    if fract == 0:
        return "0"
    elif x == 1:
        return '$R_{j}$'
    else:
        if fract.denominator == 1:
            return f'${fract.numerator}R_{{j}}$'
        else:
            return f'$\\frac{{{fract.numerator}}}{{{fract.denominator}}}$' + '$R_{j}$'
def format_fraction_with_r_jup_small(x, pos):
    fract = Fraction(x).limit_denominator()
    if fract == 0:
        return "0"
    elif x == 1:
        return '$R_{j}$'
    elif x == -1:
        return '$-R_{j}$'
    else:
        if fract.numerator == 1:
            return f'$R_{{j}}/{fract.denominator}$'
        elif fract.denominator == 1:
            return f'${fract.numerator}R_{{j}}$'
        elif fract.numerator > 0:
            return f'${fract.numerator}R_{{j}}/{fract.denominator}$'
        else:
            return f'$-{abs(fract.numerator)}R_{{j}}/{fract.denominator}$'

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
    ellipse_sign = np.sign(sin_phi)
    sin_phi *= (-1 + 2 * (
                cos_phi >= 0))  # aligns everything with closest axis instead of same axis everytime - keeps bounds w. correct sign
    cos_phi = np.abs(cos_phi)

    def find_distance_from_ellipse_centre(a, b):
        with np.errstate(all='raise'):
            if mu == 0:
                return a ** 2 + b ** 2
            else:
                return (a * cos_phi + b * sin_phi) ** 2 + (1 / mu ** 2) * (a * sin_phi - b * cos_phi) ** 2

    angles = np.linspace(0, 2 * np.pi, 5000)
    xs = r_circle * np.cos(angles) + offset
    ys = r_circle * np.sin(angles)
    in_ellipse = (find_distance_from_ellipse_centre(xs, ys) < r_ellipse ** 2)
    intersect_bool = np.roll(in_ellipse, 1) != in_ellipse
    x = xs[intersect_bool]
    y = ys[intersect_bool]

    x_prime = (x * cos_phi + y * sin_phi)
    y_prime = (y * cos_phi - x * sin_phi) / mu

    ellipse_area = np.pi * mu * r_ellipse ** 2
    circle_area = np.pi * r_circle ** 2

    if len(x) == 0:
        if np.all(in_ellipse):
            return circle_area
        elif not np.any(in_ellipse) and np.abs(offset) < r_circle:
            return ellipse_area
        else:
            return 0.

    elif len(x) == 2:

        circle_rot_angle = np.arctan((x[1] - x[0]) / (y[1] - y[0]))
        circle_bound, extra = (x - offset) * np.cos(circle_rot_angle) - y * np.sin(circle_rot_angle)

        # catching numerical errors:
        if np.abs(circle_bound / r_circle) >= 1:
            if np.sign(circle_bound == np.sign(offset)):
                return min(ellipse_area, circle_area)
            else:
                return 0.

        circle_sign = np.sign(offset) * np.sign(offset - (x[0] - y[0] * (x[1] - x[0]) / (y[1] - y[0])))
        circle_section_area = np.abs(
            circle_section_integral(r_circle, bounds=[circle_sign * np.abs(circle_bound), r_circle]))

        ellipse_rot_angle = np.arctan2((x_prime[1] - x_prime[0]), (y_prime[1] - y_prime[0]))

        ellipse_bound, extra = x_prime * np.cos(ellipse_rot_angle) - y_prime * np.sin(ellipse_rot_angle)
        # catching numerical errors:
        if np.abs(ellipse_bound / r_ellipse) >= 1:
            if np.sign(ellipse_bound) != np.sign(offset):
                return min(ellipse_area, circle_area)
            else:
                return 0.

        ellipse_sign = np.sign(offset) * np.sign(x[0] - y[0] * (x[1] - x[0]) / (y[1] - y[0]))
        ellipse_section_area = mu * np.abs(
            circle_section_integral(r_ellipse, bounds=[ellipse_sign * np.abs(ellipse_bound), r_ellipse]))
        return ellipse_section_area + circle_section_area

    elif len(x) == 4:
        intersect_index = np.where(intersect_bool)[0]
        i_0 = intersect_index[0]
        # setting up which 2 points to take each time
        # we want two points where the section of ellipse is outside the section of circle
        clockwise_connected = in_ellipse[
            i_0 - 1]  # is the first point in the list paired with the one clockwise or anti-clockwise relative to it
        connection_direction = 1 - 2 * clockwise_connected  # the sign becomes important later when finding the length of the circle section between two points
        if clockwise_connected:
            x = np.roll(x, 1)
            y = np.roll(y, 1)
            x_prime = np.roll(x_prime, 1)
            y_prime = np.roll(y_prime, 1)
            intersect_index = np.roll(intersect_index, 1)

        area_diff = 0
        for i in range(2):
            # selecting the correct pairs of intersection points
            x_i = x[2 * i:2 * (i + 1)]
            y_i = y[2 * i:2 * (i + 1)]
            x_i_prime = x_prime[2 * i:2 * (i + 1)]
            y_i_prime = y_prime[2 * i:2 * (i + 1)]
            index_i = intersect_index[2 * i:2 * (i + 1)]

            # check if the section of the circle subtending the two intersection points is less than the circumference
            # if yes the signs of the two bounds in the integral should be the same
            index_i = index_i[::connection_direction]
            if index_i[0] < index_i[1] and clockwise_connected:
                # python can't loop back around to the start when splicing arrays, or at least I don't know how
                circle_section_size = len(in_ellipse) - index_i[1] + index_i[0]
            else:
                circle_section_points = in_ellipse[index_i[0]:index_i[1]:connection_direction]
                circle_section_size = np.sum(circle_section_points)
            small_circle_section = circle_section_size < len(in_ellipse) / 2
            circle_bound_sign = 2 * small_circle_section - 1

            circle_rot_angle = np.arctan((x_i[1] - x_i[0]) / (y_i[1] - y_i[0]))
            circle_bound, extra = (x_i - offset) * np.cos(circle_rot_angle) - y_i * np.sin(circle_rot_angle)
            circle_section_area = np.abs(
                circle_section_integral(r_circle, bounds=[circle_bound_sign * np.abs(circle_bound), r_circle]))

            other_x_prime = x_prime[2 - 2 * i:int(4 / (i + 1))]
            if np.all(other_x_prime < x_i_prime):
                ellipse_radius_sign = 1
            elif np.all(other_x_prime > x_i_prime):
                ellipse_radius_sign = -1
            else:
                print('bad result')

            ellipse_rot_angle = np.arctan((x_i_prime[1] - x_i_prime[0]) / (y_i_prime[1] - y_i_prime[0]))
            ellipse_bound, extra = x_i_prime * np.cos(ellipse_rot_angle) - y_i_prime * np.sin(ellipse_rot_angle)
            ellipse_section_area = mu * np.abs(
                circle_section_integral(r_ellipse, bounds=[ellipse_bound, ellipse_radius_sign * r_ellipse]))

            area_diff -= ellipse_section_area
            area_diff += circle_section_area

        return ellipse_area + area_diff

    else:
        raise NotImplementedError('Edge case of %.i intersection points, how did you even do this??!' % len(x))


def generate_plot_style():
    plt.style.use('the_usual.mplstyle')
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(FuncFormatter(format_fraction_with_pi))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
    ax.set_xlabel(r'Phase angle $\alpha$')
    ax.set_ylabel(r'Intensity ($L_{\odot}$)')
    fig.tight_layout()
    return fig, ax

def select_best_result(results):
    lowest_NLL = np.inf
    best_fit = None
    for result in results:
        NLL = result[0]
        fit = result[1]
        if NLL < lowest_NLL:
            best_fit = fit
            lowest_NLL = NLL
    return lowest_NLL, best_fit

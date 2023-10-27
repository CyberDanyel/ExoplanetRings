# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:46:48 2023

@author: victo
"""
import time
from fractions import Fraction
import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
import matplotlib.ticker as tck


# exoring_functions


#riemann integration
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
            if old_total_integral == 0 or abs((new_total_integral - old_total_integral) / old_total_integral) < sigma or old_total_integral < 1e-10:
                return new_total_integral
            else:
                n = n * 2
                width = int(np.sqrt(n))
                iteration += 1
                old_total_integral = new_total_integral
                pass

#newton_raphson
def newton_raphson(func, x_0, tol):
    '''
    Newton-Raphson method for finding roots

    Parameters
    ----------
        func : function/lambda
    The function whose roots are being sought
        x_0 : float
    An initial guess for the root
        tol : float
    The tolerance at which to return

    Returns
    -------
    The roots of the function

    '''
    delta_x = 1e-4
    x = x_0
    err = tol + 1
    while np.abs(err) > tol:
        f_prime = (func(x+delta_x) - func(x))/(delta_x)
        err = func(x)/f_prime
        x -= err
    return x

def find_ellipse_intersection(r_0, r_1, mu, sin_phi, cos_phi, offset):
    'r_0 - radius of circle'
    'r_1 - semimajor axis of ellipse'
    'mu - ratio of semiminor axis to semimajor axis'
    t4_coeff = (cos_phi - 2*offset*cos_phi/r_0 + (offset/r_0)**2*cos_phi) + (1/mu**2) * sin_phi**2 * (1 - 2*offset/r_0 + (offset/r_0)**2) - (r_1/r_0)**2
    t3_coeff = (1/mu**2)*(4*sin_phi*cos_phi)*(offset/r_0 - 1)
    t2_coeff = (-2*cos_phi + 2*(offset/r_0)**2*cos_phi) + (1/mu**2) * (4*cos_phi**2 + sin_phi**2 * 2 * (offset/r_0)**2) - 2*(r_1/r_0)**2
    t1_coeff = (1/mu**2) * (4*sin_phi*cos_phi*(1+offset/r_0) - 2*sin_phi**2)
    t0_coeff = (cos_phi + 2*(offset/r_0)*cos_phi + (offset/r_0)**2*cos_phi) + (1/mu**2) * sin_phi**2 * (1 + 2*(offset/r_0) + (offset/r_0)**2) - (r_1/r_0)**2
            
            
    t = np.roots([t4_coeff, t3_coeff, t2_coeff, t1_coeff, t0_coeff])
    t = np.real(t[t == np.real(t)])
    y = (1-t**2)/(1+t**2)
    z = (2*t)/(1+t**2)
    
    return y, z
    
#monte carlo integration
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


#animation functions
class Animation:
    def __init__(self, planet, star, ring):
        self.star = star
        self.planet = planet
        self.ring = ring
        self.alphas = np.array(list(np.linspace(-np.pi, -0.1, 1000)) + list(np.linspace(-0.1, 0.1, 2000)) + list(
            np.linspace(0.1, np.pi, 1000)))
        self.planet_curve = None
        self.ring_curve = None
        self.maximum_intensity = None
        self.calculate_light_curves()

    def set_axes_equal(self, ax):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def generate_sphere_coords(self, centre, sphere_radius, sampling_num):
        theta = np.radians(np.linspace(0, 180, sampling_num, endpoint=True))
        phi = np.radians(np.linspace(0, 360, sampling_num, endpoint=True))
        theta_coords, phi_coords = np.meshgrid(theta, phi)
        x_coords = sphere_radius * np.sin(theta_coords) * np.cos(phi_coords) + centre[0]  # Offset by centre of sphere
        y_coords = sphere_radius * np.sin(theta_coords) * np.sin(phi_coords) + centre[1]  # Offset by centre of sphere
        z_coords = sphere_radius * np.cos(theta_coords) + centre[2]  # Offset by centre of sphere
        return x_coords, y_coords, z_coords

    def calculate_light_curves(self):
        # fig, ax = plt.subplots()
        print('started')
        a = time.time()
        self.planet_curve = self.planet.light_curve(self.alphas)
        self.ring_curve = self.ring.light_curve(self.alphas)
        b = time.time()
        self.maximum_intensity = max(self.planet_curve + self.ring_curve)
        print(f'ended in ' + str(b - a))

    def generate_animation(self):
        plt.style.use('the_usual.mplstyle')
        fig = plt.figure(figsize=plt.figaspect(2.))
        axs1 = fig.add_subplot(3, 1, 1)
        axs2 = fig.add_subplot(3, 1, (2, 3), projection='3d')
        fig.tight_layout()

        def create_graph_layout():
            axs1.xaxis.set_major_formatter(FuncFormatter(format_fraction_with_pi))
            axs1.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
            axs1.set_xlim(-1, 1)
            axs1.set_ylim(0, 1.1 * self.maximum_intensity)
            axs1.set_xlabel(r'Phase angle $\alpha$')
            axs1.set_ylabel(r'Intensity ($L_{\odot}$)')
            axs2.set_xlim(-(self.star.distance + self.star.radius + self.planet.radius),
                          self.star.distance + self.star.radius + self.planet.radius)
            axs2.set_ylim(-(self.star.distance + self.star.radius + self.planet.radius),
                          self.star.distance + self.star.radius + self.planet.radius)
            axs2.set_zlim(-(self.star.radius + self.planet.radius),
                          self.star.radius + self.planet.radius)
            axs2.set_box_aspect([10, 10, 10])
            axs2.view_init(elev=0, azim=0)  # Could make the camera centre around the planet, so we see it from closer
            self.set_axes_equal(axs2)

        def init():
            create_graph_layout()
            blue_patch = mpatches.Patch(color='blue', label='Planet')
            red_patch = mpatches.Patch(color='red', label='Ring')
            orange_patch = mpatches.Patch(color='orange', label='Planet + Ring')
            axs1.legend(handles=[blue_patch, red_patch, orange_patch], fontsize=6)

        num_frames = 100
        orbital_centre = [0, 0, 0]
        z = 0

        def update(frame):
            print('Frame', str(frame))
            axs2.clear()
            create_graph_layout()
            num_points = int(frame * len(self.alphas) / num_frames)
            axs1.plot(self.alphas[:num_points] / np.pi, self.planet_curve[:num_points], label='Planet', color='blue')
            axs1.plot(self.alphas[:num_points] / np.pi, self.ring_curve[:num_points], label='Ring', color='red')
            axs1.plot(self.alphas[:num_points] / np.pi, self.planet_curve[:num_points] + self.ring_curve[:num_points],
                      label='Planet + Ring', color='orange')
            if frame == 0:
                axs1.legend(fontsize=6)
            alpha = self.alphas[num_points]
            centre = [self.star.distance * np.cos(alpha - np.pi), self.star.distance * np.sin(alpha - np.pi),
                      z]  # np.pi factor corrects for the beginning of the phase
            star_x_coords, star_y_coords, star_z_coords = self.generate_sphere_coords([0, 0, 0],
                                                                                      sphere_radius=self.star.radius,
                                                                                      sampling_num=100)
            x_coords, y_coords, z_coords = self.generate_sphere_coords(centre + orbital_centre,
                                                                       sphere_radius=self.planet.radius,
                                                                       sampling_num=100)
            axs2.plot_surface(
                star_x_coords, star_y_coords, star_z_coords, color='orange',
                linewidth=0, antialiased=False, rstride=1, cstride=1, alpha=1)
            axs2.plot_surface(
                x_coords, y_coords, z_coords, color='blue',
                linewidth=0, antialiased=False, rstride=1, cstride=1, alpha=1)

        ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False)
        ani.save('gifs/animated_graph.gif', writer='pillow',
                 fps=20)  # Adjust the filename and frames per second as needed

def circle_section_integral(radius, bounds: []):
    upper = radius ** 2 * np.arcsin(bounds[1] / radius) + bounds[1] * np.sqrt(radius ** 2 - bounds[1] ** 2)
    bottom = radius ** 2 * np.arcsin(bounds[0] / radius) + bounds[0] * np.sqrt(radius ** 2 - bounds[0] ** 2)
    integration_result = upper - bottom
    return integration_result

def overlap_area(r_circle, r_ellipse, mu, cos_phi, sin_phi, offset):
    def find_distance_from_ellipse_centre(a, b):
        return (a * cos_phi + b * sin_phi) ** 2 + (1 / mu ** 2) * (-a * sin_phi + b * cos_phi) ** 2
    
    angles = np.linspace(0, 2*np.pi, 2000)
    xs = r_circle*np.cos(angles) + offset
    ys = r_circle*np.sin(angles)
    in_ellipse = (find_distance_from_ellipse_centre(xs, ys) < r_ellipse**2)
    x = xs[np.roll(in_ellipse, 1) != in_ellipse]
    y = ys[np.roll(in_ellipse, 1) != in_ellipse]
            
    if len(x) == 0:
        if np.abs(offset) < r_circle:
            return np.pi*mu*r_ellipse**2
        else:
            return 0.
        
            
    x_prime = (x*cos_phi - y*sin_phi)
    y_prime = (y*cos_phi + x*sin_phi)/mu
            #plt.scatter(y, z)
            #plt.scatter(y_prime, z_prime)
            
            
    circle_rot_angle = np.arctan((x[1]-x[0])/(y[1]-y[0]))
            
    x_circle_prime = (x-offset)*np.cos(circle_rot_angle) - y*np.sin(circle_rot_angle)
    #y_circle_prime = (x-offset)*np.sin(circle_rot_angle) + y*np.cos(circle_rot_angle)
           
    circle_section_area = np.abs(circle_section_integral(r_circle, bounds = [x_circle_prime[0], - np.sign(offset)*r_circle]))
            
    ellipse_rot_angle = np.arctan((x_prime[1]-x_prime[0])/(y_prime[1]-y_prime[0]))
    x_ellipse_prime = x_prime*np.cos(ellipse_rot_angle) - y_prime*np.sin(ellipse_rot_angle)
    #y_ellipse_prime = x_prime*np.sin(ellipse_rot_angle) + y_prime*np.cos(ellipse_rot_angle)
            
    ellipse_section_area = mu * np.abs(circle_section_integral(r_ellipse, bounds = [x_ellipse_prime[0], np.sign(offset)*r_ellipse]))
    return ellipse_section_area + circle_section_area

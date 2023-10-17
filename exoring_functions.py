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

import matplotlib.ticker as tck


# exoring_functions

def integrate2d(func, bounds, sigma=0.01):
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


class Integrals:
    def __init__(self, i):
        self.i = i
        xs = np.random.uniform(0, 1, self.i)
        self.thetas = np.arccos(1 - 2 * xs)
        # self.phis = np.random.uniform(- np.pi / 2, np.pi / 2,self.i)

    def monte_carlo_integration(self, alpha, func):
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


class Animation:
    def __init__(self, planet, star, ring):
        self.star = star
        self.planet = planet
        self.ring = ring
        self.alphas = np.linspace(-np.pi, np.pi, 1000)
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

    def format_fraction_with_pi(self, x, pos):
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

        def init():
            axs1.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
            axs1.xaxis.set_major_formatter(FuncFormatter(self.format_fraction_with_pi))
            axs1.set_ylim(0, 1.1 * self.maximum_intensity)
            axs1.set_xlim(-1, 1)
            axs1.set_xlabel(r'Phase angle $\alpha$')
            axs1.set_ylabel(r'Intensity (arbitrary)')
            axs2.set_xlim(-(self.star.distance + self.star.radius + self.planet.radius),
                          self.star.distance + self.star.radius + self.planet.radius)
            axs2.set_ylim(-(self.star.distance + self.star.radius + self.planet.radius),
                          self.star.distance + self.star.radius + self.planet.radius)
            axs2.set_zlim(-(self.star.distance + self.star.radius + self.planet.radius),
                          self.star.distance + self.star.radius + self.planet.radius)
            axs2.set_box_aspect([10, 10, 10])
            axs2.view_init(elev=0, azim=0)
            self.set_axes_equal(axs2)

        num_frames = 100
        orbital_centre = [0, 0, 0]
        z = 0

        def update(frame):
            fig.tight_layout()
            print('Frame', str(frame))
            axs2.clear()
            axs2.set_xlim(-(self.star.distance + self.star.radius + self.planet.radius),
                          self.star.distance + self.star.radius + self.planet.radius)
            axs2.set_ylim(-(self.star.distance + self.star.radius + self.planet.radius),
                          self.star.distance + self.star.radius + self.planet.radius)
            axs2.set_zlim(-(self.star.distance + self.star.radius + self.planet.radius),
                          self.star.distance + self.star.radius + self.planet.radius)
            axs2.set_box_aspect([10, 10, 10])
            axs2.view_init(elev=0, azim=0)
            self.set_axes_equal(axs2)
            num_points = int(frame * len(self.alphas) / num_frames)
            alpha = self.alphas[num_points]
            axs1.plot(self.alphas[:num_points] / np.pi, self.planet_curve[:num_points], label='Planet', color='blue')
            axs1.plot(self.alphas[:num_points] / np.pi, self.ring_curve[:num_points], label='Ring', color='red')
            axs1.plot(self.alphas[:num_points] / np.pi, self.planet_curve[:num_points] + self.ring_curve[:num_points],
                      label='Ring + Planet', color='orange')
            axs1.xaxis.set_major_formatter(FuncFormatter(self.format_fraction_with_pi))
            axs1.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
            axs1.set_xlim(-1, 1)
            axs1.set_ylim(0, 1.1 * self.maximum_intensity)
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
            # Add more lines if needed

        axs1.set_xlabel(r'Phase angle $\alpha$')
        axs1.set_ylabel(r'Intensity (arbitrary)')
        axs1.legend()
        ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False)
        ani.save('gifs/animated_graph.gif', writer='pillow',
                 fps=10)  # Adjust the filename and frames per second as needed

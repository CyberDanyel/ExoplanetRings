import matplotlib.pyplot as plt
import time
import numpy as np
import exoring_objects
import exoring_functions
import matplotlib.ticker as tck
import matplotlib.patches as mpatches
import matplotlib.animation as animation_package
from matplotlib.ticker import FuncFormatter


class Animation:
    def __init__(self, planet, star, ring, including_star=False):
        self.star = star
        self.planet = planet
        self.ring = ring
        self.including_star = including_star
        self.maximum_intensity_including_star = None
        self.maximum_intensity_excluding_star = None
        self.alphas = np.array(list(np.linspace(-np.pi, -0.1, 1000)) + list(np.linspace(-0.1, 0.1, 2000)) + list(
            np.linspace(0.1, np.pi, 1000)))
        self.planet_curve = None
        self.ring_curve = None
        self.star_curve = None
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
        self.star_curve = self.star.light_curve(self.alphas)
        b = time.time()
        if self.including_star:
            self.maximum_intensity_including_star = max(self.planet_curve + self.ring_curve + self.star_curve)
        self.maximum_intensity_excluding_star = max(self.planet_curve + self.ring_curve)
        print(f'ended in ' + str(b - a))

    def generate_animation(self):
        plt.style.use('the_usual.mplstyle')
        if self.including_star:
            fig = plt.figure(figsize=plt.figaspect(3.))
            axs1 = fig.add_subplot(4, 1, 1)
            axs3 = fig.add_subplot(4, 1, 2)
            axs2 = fig.add_subplot(4, 1, (3, 4), projection='3d')
            fig.tight_layout()
        else:
            fig = plt.figure(figsize=plt.figaspect(2.))
            axs1 = fig.add_subplot(3, 1, 1)
            axs2 = fig.add_subplot(3, 1, (2, 3), projection='3d')
            fig.tight_layout()

        def create_graph_layout():
            axs1.xaxis.set_major_formatter(FuncFormatter(exoring_functions.format_fraction_with_pi))
            axs1.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
            axs1.set_xlim(-1, 1)
            axs1.set_ylim(0, 1.1 * self.maximum_intensity_excluding_star)
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
            if self.including_star:
                axs3.xaxis.set_major_formatter(FuncFormatter(exoring_functions.format_fraction_with_pi))
                axs3.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
                axs3.set_xlim(-1, 1)
                axs3.set_ylim(0, 1.1 * self.maximum_intensity_including_star)
                axs3.set_xlabel(r'Phase angle $\alpha$')
                axs3.set_ylabel(r'Intensity ($L_{\odot}$)')

        def init():
            create_graph_layout()
            blue_patch = mpatches.Patch(color='blue', label='Planet')
            red_patch = mpatches.Patch(color='red', label='Ring')
            orange_patch = mpatches.Patch(color='orange', label='Planet + Ring')
            axs1.legend(handles=[blue_patch, red_patch, orange_patch], fontsize=6)
            if self.including_star:
                orange_patch2 = mpatches.Patch(color='orange', label='Planet + Ring + Star')
                axs3.legend(handles=[orange_patch2], fontsize=6)

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
            if self.including_star:
                axs3.plot(self.alphas[:num_points] / np.pi,
                          self.planet_curve[:num_points] + self.ring_curve[:num_points] + self.star_curve[:num_points],
                          label='Planet + Ring + Star', color='orange')
                if frame == 0:
                    axs3.legend(fontsize=6)
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
            fig.tight_layout()

        ani = animation_package.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False)
        ani.save('gifs/animated_graph.gif', writer='pillow',
                 fps=20)  # Adjust the filename and frames per second as needed


AU = 1.495978707e13
L_SUN = 3.828e33
R_JUP = 6.9911e9
R_SUN = 6.957e10
JUP_TO_AU = AU / R_JUP
SUN_TO_JUP = R_SUN / R_JUP
SUN_TO_AU = AU / R_SUN

plt.style.use('the_usual.mplstyle')

star = exoring_objects.Star(1, 1 * SUN_TO_JUP, .1 * JUP_TO_AU, 1)

planet = exoring_objects.Planet(0.52, 1, star)

ring_normal = np.array([1., 1., 0.])
ring_normal /= np.sqrt(np.sum(ring_normal * ring_normal))

ring_normal2 = np.array([1., 0., 0.0])
ring_normal2 /= np.sqrt(np.sum(ring_normal * ring_normal))

ring = exoring_objects.Ring(0.7, 1, 2., ring_normal, star)
star.planet = planet  # Should use inheritance to prevent this being necessary
# ring2 = Ring(0.8, 1, 10, ring_normal2, star)

animation = Animation(planet, star, ring, including_star=True)
animation.generate_animation()

plt.style.use('the_usual.mplstyle')
# plt.subplots_adjust(top=2.1, bottom=2, tight_layout=True)
fig, ax = plt.subplots()
ax.plot(animation.alphas / np.pi, animation.planet_curve, label='Planet')
ax.plot(animation.alphas / np.pi, animation.ring_curve, label='Ring')
ax.plot(animation.alphas / np.pi, animation.planet_curve + animation.ring_curve, label='Ring + Planet')
ax.xaxis.set_major_formatter(FuncFormatter(exoring_functions.format_fraction_with_pi))
ax.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
ax.set_xlabel(r'Phase angle $\alpha$')
ax.set_ylabel(r'Intensity ($L_{\odot}$)')
ax.legend()
plt.tight_layout()
fig.savefig('images/light_curves.jpg', bbox_inches="tight")

fig, ax = plt.subplots()
ax.plot(animation.alphas / np.pi, animation.planet_curve + animation.ring_curve + animation.star_curve,
        label='Ring + Planet + Star')
ax.xaxis.set_major_formatter(FuncFormatter(exoring_functions.format_fraction_with_pi))
ax.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
ax.set_xlabel(r'Phase angle $\alpha$')
ax.set_ylabel(r'Intensity ($L_{\odot}$)')
ax.legend()
plt.tight_layout()
fig.savefig('images/light_curves_with_star.jpg', bbox_inches="tight")

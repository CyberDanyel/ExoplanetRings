# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:11:07 2023

@author: daniel
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def set_axes_equal(ax):
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
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

sphere_radius = 1
def limb_darkening_gaussian(x,y):
    radius = np.sqrt(x**2+y**2)
    return math.e**(-(radius**(2))/(0.5))

def generate_sphere_coords(centre,sphere_radius,sampling_num):
    theta = np.radians(np.linspace(0,180,sampling_num,endpoint=True))
    phi = np.radians(np.linspace(0,360,sampling_num,endpoint=True))
    theta_coords,phi_coords = np.meshgrid(theta,phi)
    x_coords = sphere_radius * np.sin(theta_coords) * np.cos(phi_coords) + centre[0] # Offset by centre of sphere
    y_coords = sphere_radius * np.sin(theta_coords) * np.sin(phi_coords) + centre[1] # Offset by centre of sphere
    z_coords = sphere_radius * np.cos(theta_coords) + centre[2] # Offset by centre of sphere
    return x_coords, y_coords, z_coords
'''
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x_coords, y_coords, z_coords = generate_sphere_coords([1,1,1], sphere_radius=1, sampling_num=500)
plot = ax.plot_surface(
    x_coords, y_coords, z_coords, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
    linewidth=0, antialiased=False, alpha=0.5)
ax.set_box_aspect([1,1,1])
set_axes_equal(ax)
plt.show()
'''
def plot_orbiting_planet(orbital_centre, orbital_radius, star_radius, frames):
    #generating frames
    alphas = np.radians(np.arange(0,360,360/frames))
    z = 0 # Setting orbit on theta = pi/2
    star_x_coords, star_y_coords, star_z_coords = generate_sphere_coords([0, 0, 0], sphere_radius=star_radius,sampling_num=100)
    for frame_number in range(frames):
        alpha = alphas[frame_number]
        centre = [orbital_radius * np.cos(alpha),orbital_radius * np.sin(alpha),z]
        x_coords, y_coords, z_coords = generate_sphere_coords(centre+orbital_centre, sphere_radius=1, sampling_num=100)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111,projection='3d')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)
        ax.set_box_aspect([10, 10, 10])
        ax.plot_surface(
            star_x_coords, star_y_coords, star_z_coords, color='orange',
            linewidth=0, antialiased=False, rstride=1, cstride=1, alpha=1)
        ax.plot_surface(
            x_coords, y_coords, z_coords, color='blue',
            linewidth=0, antialiased=False, rstride=1, cstride=1, alpha=1)
        set_axes_equal(ax)
        ax.view_init(elev=0, azim=0)
        plt.savefig(f"images/{frame_number}.png")
        plt.close()

    from PIL import Image
    images = [Image.open(f"images/{frame_number}.png") for frame_number in range(frames)]
    images[0].save('gifs/orbit.gif', save_all=True, append_images=images[1:], duration=200, loop=0)

plot_orbiting_planet([0,0,0],7,2, 50)
'''
brightness_values = limb_darkening_gaussian(x_coords,y_coords)
fig,ax=plt.subplots()
cp = ax.contourf(x_coords, y_coords, brightness_values,cmap = 'Greys_r',levels=1000,vmin=0,vmax=1)
cbar = fig.colorbar(cp) # Add a colorbar to a plot
cbar.ax.tick_params(labelsize=12) 
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.axis('equal')
plt.show()
'''
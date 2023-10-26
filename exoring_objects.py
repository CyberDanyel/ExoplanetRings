# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:51:16 2023

@author: victo
"""
# exoring objects

import numpy as np
import exoring_functions
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.ticker import FuncFormatter

# for debugging purposes

AU = 1.495978707e13
L_SUN = 3.828e33
R_JUP = 6.9911e9
R_SUN = 6.957e10
JUP_TO_AU = AU / R_JUP
SUN_TO_JUP = R_SUN / R_JUP
SUN_TO_AU = AU / R_SUN


# coordinate systems defined such that the observer is always along the x-axis
# planet always at origin
# phase angle alpha is aligned with the phi of the spherical coordinate system; the star is always 'equatorial'

# everything also assumes circular orbits


class Planet:
    def __init__(self, albedo, radius, star):
        """
        Parameters
        ----------
            radius : float
        The radius of the sphere.
        """
        self.radius = radius
        self.sc_law = lambda \
                mu_star: albedo / np.pi  # isotropic scattering law intensity distribution - 1/pi factor from
        # normalization - currently a function to future-proof
        self.phase_curve = np.vectorize(
            self.phase_curve_unvectorized)  # vectorizing so that arrays of phase angles can be input more
        # efficiently than with a Python for loop
        self.star = star
        # self.MonteCarloPlanetIntegration = exoring_functions.MonteCarloPlanetIntegration(100000)
        # this definition of phase curve includes albedo already

    def get_mu_star(self, theta, phi, alpha):
        """
        Parameters
        ----------
            theta : theta in local spherical coordinate system - observer and star always at theta = pi/2
            phi : phi in local spherical coordinate system - aligned w. phase angle
            alpha: phase angle

        Returns
        -------
            mu_star = cos(angle between star and normal to surface at these coords)
        """
        return np.sin(theta) * (np.cos(alpha) * np.cos(phi) + np.sin(alpha) * np.sin(phi))  # see notes for derivation

    def get_mu(self, theta, phi):
        """
        Parameters
        ----------
            theta : angle theta in local spherical coordinate system - observer and star always at theta = pi/2
            phi : angle phi in local spherical coordinate system - aligned w. phase angle

        Returns
        -------
            mu = cos(angle between observer and normal to surface at these coords)
        """
        return np.sin(theta) * np.cos(phi)  # see notes for derivation

    def phase_curve_integrand(self, theta, phi, alpha):
        """
        Parameters
        ----------
            theta : angle theta in local spherical coord system - see get_mu()
            phi : angle phi in local spherical coord system - see get_mu()
            alpha : phase angle
        Returns
        -------
        The integrand for the phase integral

        """
        mu = self.get_mu(theta, phi)
        mu_star = self.get_mu_star(theta, phi, alpha)
        return np.sin(theta) * mu * mu_star * self.sc_law(mu_star) * self.secondary_eclipse(theta, phi,
                                                                                            alpha)  # * (np.abs(
        # alpha) >= np.arcsin((self.star.radius + self.radius) / self.star.distance))
        # Only check secondary eclipse for certain alphas when you are close to the star for speed

    def phase_curve_unvectorized(self, alpha: float) -> object:
        """
        Parameters
        ----------
            alpha : phase angle
        Returns
        -------
        The phase curve evaluated at the phase angle alpha
        """
        # return spi.nquad(lambda theta, phi: self.phase_curve_integrand(theta, phi, alpha), ranges = [[0, np.pi],
        # [max(alpha-np.pi/2, -np.pi/2), min(alpha + np.pi/2, np.pi/2)]])[0]
        return exoring_functions.integrate2d(lambda theta, phi: self.phase_curve_integrand(theta, phi, alpha),
                                             bounds=[[0, np.pi], [max(alpha - np.pi / 2, -np.pi / 2),
                                                                  min(alpha + np.pi / 2, np.pi / 2)]], sigma=1e-3)
        # return self.MonteCarloPlanetIntegration.integrate(alpha,lambda theta, phi: self.phase_curve_integrand(theta,phi, alpha)) # Monte Carlo
        # the lambda allows for integration across two variables while the alpha is kept
        # constant within the method

    def secondary_eclipse(self, theta, phi, alpha):
        """returns boolean of whether these coords are eclipsed at this phase angle"""
        if np.abs(alpha) > 2.1 * self.star.radius / self.star.distance:
            return 1.
        return ((self.radius * np.sin(theta) * np.sin(phi) - self.star.distance * np.sin(alpha)) ** 2 + (
                self.radius * np.cos(theta)) ** 2 > self.star.radius ** 2)

    def light_curve(self, alpha):
        """turns a phase curve into a light curve"""
        return self.radius ** 2 * self.star.luminosity * (1 / (4 * self.star.distance ** 2)) * self.phase_curve(alpha)


class Ring:
    def __init__(self, albedo, inner_rad, outer_rad, normal, star):
        self.inner_radius = inner_rad
        self.outer_radius = outer_rad
        self.sc_law = lambda \
                mu_star: albedo / np.pi  # isotropic scattering law intensity distribution - 1/pi factor from
        # normalization
        self.normal = normal
        self.secondary_eclipse = np.vectorize(self.analytic_secondary_eclipse)
        self.star = star

    def get_mu_star(self, alpha):
        """mu_star = cos(angle between star and normal to ring)"""
        star_pos = np.array([np.cos(alpha), np.sin(alpha), 0.])
        return np.dot(self.normal, star_pos)

    def get_mu(self):
        """mu = cos(angle between observer and normal to ring)"""
        obs_pos = np.array([1, 0, 0])
        return np.dot(self.normal, obs_pos)

    def phase_curve(self, alpha):
        """phase curve innit"""
        mu = self.get_mu()
        mu_star = self.get_mu_star(alpha)
        return mu * mu_star * self.sc_law(mu_star) * (mu_star > 0) * self.secondary_eclipse(alpha)  # boolean prevents
        # forwards scattering

    def unvectorized_secondary_eclipse(self, alpha):
        """finds the amount of flux to subtract from the ring - since there is no integral for the total ring
        scattering"""

        if np.abs(alpha) > 2.1 * self.star.radius / self.star.distance:
            return 1.

        mu = self.get_mu()
        n_x, n_y, n_z = self.normal

        y_star = self.star.distance * np.sin(alpha)
        z_star = 0.

        # bounds_z = [max(-self.outer_radius, -self.star.radius), min(self.outer_radius, self.star.radius)] bounds_y
        # = [max(-self.outer_radius, y_star - self.star.radius), min(self.outer_radius, y_star + self.star.radius)]
        bounds_z = [-self.outer_radius, self.outer_radius]
        bounds_y = [-self.outer_radius, self.outer_radius]
        sin_theta = np.sqrt(1 - mu ** 2)
        cos_phi = n_z / sin_theta
        sin_phi = n_y / sin_theta

        def find_distance_from_ring_centre(y, z):
            return np.sqrt(
                (y * cos_phi + z * sin_phi) ** 2 + (1 / mu ** 2) * (-y * sin_phi + z * cos_phi) ** 2)

        def outside_inner_radius(y, z):
            return find_distance_from_ring_centre(y, z) > self.inner_radius

        def inside_outer_radius(y, z):
            return find_distance_from_ring_centre(y, z) < self.outer_radius

        def in_shadow(y, z):
            return (y - y_star) ** 2 + (z - z_star) ** 2 < self.star.radius ** 2

        numerator = exoring_functions.integrate2d(
            lambda y, z: outside_inner_radius(y, z) * inside_outer_radius(y, z) * in_shadow(y, z),
            [bounds_y, bounds_z])
        denominator = exoring_functions.integrate2d(
            lambda y, z: outside_inner_radius(y, z) * inside_outer_radius(y, z),
            [bounds_y, bounds_z])  # - self.inner_radius**2)#*mu_star
        # numerator = exoring_functions.monte_carlo_ring_integration(
        #    lambda y, z: outside_inner_radius(y, z) * inside_outer_radius(y, z) * in_shadow(y, z),
        #    bounds_y, bounds_z, 10000)
        # denominator = exoring_functions.monte_carlo_ring_integration(
        #    lambda y, z: outside_inner_radius(y, z) * inside_outer_radius(y, z),
        #    bounds_y, bounds_z, 10000)
        # print(abs(1 - numerator / denominator)) # numerical errors may bring this down to 0
        return abs(1 - numerator / denominator)  # numerical errors may bring this down to 0
    
    def analytic_secondary_eclipse(self, alpha):
        if np.abs(alpha) > 4.0 * self.star.radius / self.star.distance:
            return 1.
        
        if np.abs(alpha) < 0.05:
            break_val = 'filler'
        
        mu = self.get_mu()
        n_x, n_y, n_z = self.normal

        y_star = self.star.distance * np.sin(alpha)
        z_star = 0.
        
        #sorry im using a negative version of phi as compared to the above
        sin_theta = np.sqrt(1 - mu ** 2)
        cos_phi = n_z / sin_theta
        sin_phi = -n_y / sin_theta
        
        
        def find_distance_from_ring_centre(y, z):
            return (y * cos_phi + z * sin_phi) ** 2 + (1 / mu ** 2) * (-y * sin_phi + z * cos_phi) ** 2


        areas = []
        for i in range(2):
            r_star = self.star.radius
            if i == 0:
                r = self.inner_radius
            else:
                r = self.outer_radius
            
            angles = np.linspace(0, 2*np.pi, 5000)
            ys = r_star*np.cos(angles) + y_star
            zs = r_star*np.sin(angles)
            in_ellipse = find_distance_from_ring_centre(ys, zs) < r**2
            y = ys[np.roll(in_ellipse, 1) != in_ellipse]
            z = zs[np.roll(in_ellipse, 1) != in_ellipse]
            
            #y, z = exoring_functions.find_ellipse_intersection(r_star, r, mu, sin_phi, cos_phi, y_star)
            
            if len(y) == 0:
                if np.abs(y_star) < np.abs(self.star.radius):
                    return 0.
                else:
                    return 1.
            
            y_prime = (y*cos_phi - z*sin_phi)
            z_prime = (z*cos_phi + y*sin_phi)/mu
            if i == 1:
                plt.scatter(y_prime, z_prime)
            
            y+=y_star
            if i == 1:
                plt.scatter(y, z)
            
            circle_angle = np.abs(np.arctan(z[1]/y[1])) + np.abs(np.arctan(z[0]/y[0]))
            circle_sector = 0.5*circle_angle*self.star.radius**2
            circle_triangle = np.cos(circle_angle/2)*np.sin(circle_angle/2)*self.star.radius**2
            circle_area = circle_sector - circle_triangle
                
            #y_prime = (y + y_star)*cos_phi - z*sin_phi
            #z_prime = z*cos_phi + (y + y_star)*sin_phi/mu
            
            
            
            ellipse_angle = np.abs(np.arctan(z_prime[1]/y_prime[1])) + np.abs(np.arctan(z_prime[0]/y_prime[0]))
            ellipse_sector = ellipse_angle*r**2
            ellipse_triangle = np.cos(ellipse_angle/2)*np.sin(ellipse_angle/2)*r**2
            ellipse_area = mu * (ellipse_sector - ellipse_triangle)
                
            total_area = circle_area + ellipse_area
            areas.append(total_area)

        area_on_ring = areas[1]-areas[0]
        area_frac = area_on_ring / (mu*np.pi*(self.outer_radius**2) - self.inner_radius**2)
        return 1-area_frac
            
        
    def light_curve(self, alpha):
        return (self.outer_radius ** 2 - self.inner_radius ** 2) * self.phase_curve(alpha) * self.star.luminosity / (
                4 * self.star.distance ** 2)


class Star:
    def __init__(self, luminosity, radius, distance, mass):
        self.luminosity = luminosity
        self.radius = radius
        self.distance = distance
        self.mass = mass


#%%%

# debugging stuff


#plt.style.use('the_usual')
star = Star(1, 1 * SUN_TO_JUP, .1 * JUP_TO_AU, 1)

planet = Planet(0.52, 1, star)

ring_normal = np.array([1., 1., 0.0])
ring_normal /= np.sqrt(np.sum(ring_normal * ring_normal))

ring_normal2 = np.array([1., 0., 0.0])
ring_normal2 /= np.sqrt(np.sum(ring_normal * ring_normal))

ring = Ring(0.7, 2, 3., ring_normal, star)
# ring2 = Ring(0.8, 1, 10, ring_normal2, star)

alphas = np.array(list(np.linspace(-np.pi, -0.1, 1000)) + list(np.linspace(-.1, .1, 3000)) + list(np.linspace(.1, np.pi, 1000)))
#planet_curve = planet.light_curve(alphas)
ring_curve = ring.light_curve(alphas)

plt.title('Physics Barbie is the best Barbie')
#plt.plot(alphas, planet_curve, label = 'Planet')
#plt.plot(alphas, ring_curve, label = 'Ring')
#plt.plot(alphas, planet_curve + ring_curve, label = r'Planet + Ring')
#plt.xlabel(r'Phase angle $\alpha$')
#plt.ylabel(r'Luminosity ($L_\odot$)')
#plt.legend()
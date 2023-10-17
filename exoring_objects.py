# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:51:16 2023

@author: victo
"""
import time

# exoring objects

import numpy as np
import scipy.integrate as spi
import exoring_functions
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from fractions import Fraction
import matplotlib.ticker as tck
#for debugging purposes


#coordinate systems defined such that the observer is always along the x-axis
#planet always at origin
#phase angle alpha is aligned with the phi of the spherical coordinate system; the star is always 'equatorial'

#everything also assumes circular orbits


    
class Planet:
    def __init__(self, albedo, radius, star):
        '''
        Parameters
        ----------
            radius : float
        The radius of the sphere.
        '''
        self.radius = radius
        self.sc_law = lambda mu_star:albedo/np.pi # isotropic scattering law intensity distribution - 1/pi factor from normalization - currently a function to future-proof
        self.phase_curve = np.vectorize(self.phase_curve_unvectorized) # vectorizing so that arrays of phase angles can be input more efficiently than with a Python for loop
        self.star = star
        self.Integrals = exoring_functions.Integrals(100000)
        # this definition of phase curve includes albedo already
        
    def get_mu_star(self, theta, phi, alpha):
        '''
        Parameters
        ----------
            theta : theta in local spherical coordinate system - observer and star always at theta = pi/2
            phi : phi in local spherical coordinate system - aligned w. phase angle
            alpha: phase angle
        
        Returns
        -------
            mu_star = cos(angle between star and normal to surface at these coords)
        '''
        return np.sin(theta)*(np.cos(alpha)*np.cos(phi) + np.sin(alpha)*np.sin(phi)) # see notes for derivation
    
    def get_mu(self, theta, phi):
        '''
        Parameters
        ----------
            theta : angle theta in local spherical coordinate system - observer and star always at theta = pi/2
            phi : agnle phi in local spherical coordinate system - aligned w. phase angle
            alpha: phase angle
        
        Returns
        -------
            mu = cos(angle between observer and normal to surface at these coords)
        '''
        return np.sin(theta)*np.cos(phi) # see notes for derivation
    
    def phase_curve_integrand(self, theta, phi, alpha):
        '''
        Parameters
        ----------
            theta : angle theta in local spherical coord system - see get_mu()
            phi : angle phi in local spherical coord system - see get_mu()
            alpha : phase angle
        Returns
        -------
        The integrand for the phase integral

        '''
        mu = self.get_mu(theta, phi)
        mu_star = self.get_mu_star(theta, phi, alpha)
        return np.sin(theta) * mu * mu_star * self.sc_law(mu_star) * self.secondary_eclipse(theta, phi, alpha)# * (np.abs(alpha) >= np.arcsin((self.star.radius + self.radius) / self.star.distance))
        #Only check secondary eclipse for certain alphas when you are close to the star for speed
    def phase_curve_unvectorized(self, alpha: object) -> object:
        '''
        Parameters
        ----------
            alpha : phase angle
        Returns
        -------
        The phase curve evaluated at the phase angle alpha
        '''
        #return spi.nquad(lambda theta, phi: self.phase_curve_integrand(theta, phi, alpha), ranges = [[0, np.pi], [max(alpha-np.pi/2, -np.pi/2), min(alpha + np.pi/2, np.pi/2)]])[0]
        return exoring_functions.integrate2d(lambda theta, phi: self.phase_curve_integrand(theta, phi, alpha), bounds = [[0, np.pi], [max(alpha-np.pi/2, -np.pi/2), min(alpha + np.pi/2, np.pi/2)]], sigma = 1e-3)
        #return self.Integrals.monte_carlo_integration(alpha,lambda theta, phi: self.phase_curve_integrand(theta, phi, alpha)) # Monte Carlo
        # the lambda allows for integration across two variables while the alpha is kept constant within the method
        
    def secondary_eclipse(self, theta, phi, alpha):
        'returns False if these coords are eclipsed at this phase angle, True otherwise'
        return ((self.radius*np.sin(theta)*np.sin(phi) - self.star.distance*np.sin(alpha))**2 + (self.radius*np.cos(theta))**2 > self.star.radius**2)
        
    def light_curve(self, alpha):
        'turns a phase curve into a light curve'
        return self.radius**2 * self.star.luminosity * (1/(4*self.star.distance**2)) * self.phase_curve(alpha)


class Ring:
    def __init__(self, albedo, inner_rad, outer_rad, normal, star):
        self.inner_radius = inner_rad
        self.outer_radius = outer_rad
        self.sc_law = lambda mu_star:albedo/np.pi # isotropic scattering law intensity distribution - 1/pi factor from normalization
        self.normal = normal
        self.star = star
        
    def get_mu_star(self, alpha):
        'mu_star = cos(angle between star and normal to ring)'
        star_pos = np.array([np.cos(alpha), np.sin(alpha), 0.])
        return np.dot(self.normal, star_pos)
    
    def get_mu(self):
        'mu = cos(angle between observer and normal to ring)'
        obs_pos = np.array([1, 0, 0])
        return np.dot(self.normal, obs_pos)
        
    def phase_curve(self, alpha):
        'phase curve innit'
        mu = self.get_mu()
        mu_star = self.get_mu_star(alpha)
        return mu * mu_star * self.sc_law(mu_star) * (mu_star > 0) # boolean prevents forwards scattering
    
    def secondary_eclipse(self, alpha):
        'finds the amount to subtract from the ring - since there is no integral'
        #struggling
    
    def light_curve(self, alpha):
        return (self.outer_radius**2-self.inner_radius**2)*self.phase_curve(alpha)*self.star.luminosity/(4*self.star.distance**2)

class Star:
    def __init__(self, luminosity, radius, distance, mass):
        self.luminosity = luminosity
        self.radius = radius
        self.distance = distance
        self.mass = mass

#%%%
#debugging stuff
star = Star(1000000, 10, 50, 10)
planet = Planet(1, 1, star)

ring_normal = np.array([1., 1., 0.0])
ring_normal /= np.sum(ring_normal*ring_normal)

ring = Ring(1, 2, 3, ring_normal, star)

animation = exoring_functions.Animation(planet,star,ring)
animation.generate_animation()

plt.style.use('the_usual.mplstyle')
plt.figure()
plt.plot(animation.alphas, animation.planet_curve, label = 'Planet')
plt.plot(animation.alphas, animation.ring_curve, label = 'Ring')
plt.plot(animation.alphas, animation.planet_curve + animation.ring_curve, label = 'Ring + Planet')

plt.xlabel(r'Phase angle $\alpha$')
plt.ylabel(r'Intensity (arbitrary)')
plt.legend()
plt.tight_layout()
plt.savefig('images/light_curves.jpg')

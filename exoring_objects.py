# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:51:16 2023

@author: victo
"""

# exoring objects

import numpy as np
import scipy.integrate as spi

import matplotlib.pyplot as plt
#for debugging purposes


#coordinate systems defined such that the observer is always along the x-axis
#planet always at origin
#phase angle alpha is aligned with the phi of the spherical coordinate system; the star is always 'equatorial'

class Planet:
    def __init__(self, albedo, radius):
        '''
        Parameters
        ----------
            radius : float
        The radius of the sphere.
        '''
        self.radius = radius
        self.sc_law = lambda mu_star:albedo/np.pi # isotropic scattering law intensity distribution - 1/pi factor from normalization - currently a function to future-proof
        self.phase_curve = np.vectorize(self.phase_curve_unvectorized) # vectorizing so that arrays of phase angles can be input more efficiently than with a Python for loop
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
        return np.sin(theta) * mu * mu_star * self.sc_law(mu_star)
    
    def phase_curve_unvectorized(self, alpha):
        '''
        Parameters
        ----------
            alpha : phase angle
        Returns
        -------
        The phase curve evaluated at the phase angle alpha
        '''
        return spi.dblquad(lambda theta, phi: self.phase_curve_integrand(theta, phi, alpha), max(alpha-np.pi/2, -np.pi/2), min(alpha + np.pi/2, np.pi/2), 0, np.pi)[0]
        # the lambda allows for integration across two variables while the alpha is kept constant within the method
        
class Ring:
    def __init__(self, albedo, inner_rad, outer_rad, normal):
        self.inner_radius = inner_rad
        self.outer_radius = outer_rad
        self.sc_law = lambda mu_star:albedo/np.pi # isotropic scattering law intensity distribution - 1/pi factor from normalization
        self.normal = normal
        
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
    
    
#%%%

#debugging stuff

plt.style.use('the_usual')

planet = Planet(1, 1)

ring_normal = np.array([1., 0.5, 0.2])
ring_normal /= np.sum(ring_normal*ring_normal)

ring = Ring(1, 2, 2.2, ring_normal)
alphas = np.linspace(-np.pi+0.001, np.pi-0.001, 100)

planet_phase_curve = planet.phase_curve(alphas)
ring_phase_curve = ring.phase_curve(alphas)

R = 10 # distance to star
L = 1 # Luminosity of star

planet_intensity = planet_phase_curve * (L*planet.radius**2/(4*R**2))
ring_intensity = ring_phase_curve * (L*(ring.outer_radius**2 - ring.inner_radius**2)/(4*R**2))


plt.plot(alphas, planet_intensity, label = 'Planet')
plt.plot(alphas, ring_intensity, label = 'Ring')
plt.plot(alphas, ring_intensity+planet_intensity, label = 'Ring+Planet')

plt.xlabel(r'Phase angle $\alpha$')
plt.ylabel(r'Intensity (arbitrary)')
plt.legend()

plt.tight_layout()


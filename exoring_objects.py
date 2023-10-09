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

class sphere():
    def __init__(self, albedo, radius):
        '''
        Parameters
        ----------
            radius : float
        The radius of the sphere.
        '''
        self.radius = radius
        self.sc_law = lambda mu_star:albedo/np.pi # isotropic scattering law intensity distribution - 1/pi factor from normalization
        self.phase_curve = np.vectorize(self.phase_curve_unvectorized) # this definition of phase curve includes albedo already
        
    def get_mu_star(self, theta, phi, alpha):
        'mu_star = cos(angle between star and normal to surface at these coords)'
        return np.sin(theta)*(np.cos(alpha)*np.cos(phi) + np.sin(alpha)*np.sin(phi))
    
    def get_mu(self, theta, phi):
        'mu = cos(angle between observer and normal to surface at these coords)'
        return np.sin(theta)*np.cos(phi)
    
    def phase_curve_integrand(self, theta, phi, alpha):
        mu = self.get_mu(theta, phi)
        mu_star = self.get_mu_star(theta, phi, alpha)
        return np.sin(theta) * mu * mu_star * self.sc_law(mu_star)
    
    def phase_curve_unvectorized(self, alpha):
        return spi.dblquad(lambda theta, phi: self.phase_curve_integrand(theta, phi, alpha), alpha-np.pi/2, np.pi/2, 0, np.pi)[0]
        
        
class ring():
    def __init__(self, albedo, inner_rad, outer_rad, normal):
        self.inner_radius = inner_rad
        self.outer_radius = outer_rad
        self.sc_law = lambda mu_star:albedo/np.pi # isotropic scattering law intensity distribution - 1/pi factor from normalization
        self.normal = normal
        
    def get_mu_star(self, alpha):
        'mu_star = cos(angle between star and normal to surface)'
        star_pos = np.array([np.cos(alpha), np.sin(alpha), 0])
        return np.dot(self.normal, star_pos)
    
    def get_mu(self):
        'mu = cos(angle between observer and normal to surface'
        obs_pos = np.array([1, 0, 0])
        return np.dot(self.normal, obs_pos)
        
    def phase_curve(self, alpha):
        mu = self.get_mu()
        mu_star = self.get_mu_star(alpha)
        return mu * mu_star * self.sc_law(mu_star) * (mu_star > 0) # boolean prevents forwards scattering
    
    
#%%%

#debugging stuff

plt.style.use('the_usual')

planet = sphere(1, 1)

ring_normal = np.array([1., 0, 0])
ring_normal /= np.sum(ring_normal*ring_normal)

r = ring(1, 2, 2.2, ring_normal)
alphas = np.linspace(0, np.pi-0.001, 100)

planet_phase_curve = planet.phase_curve(alphas)
ring_phase_curve = r.phase_curve(alphas)

R = 10 # distance to star
L = 1 # Luminosity of star

planet_intensity = planet_phase_curve * (L*planet.radius**2/(4*R**2))
ring_intensity = ring_phase_curve * (L*(r.outer_radius**2 - r.inner_radius**2)/(4*R**2))


plt.plot(alphas, planet_intensity, label = 'Planet')
plt.plot(alphas, ring_intensity, label = 'Ring')
plt.plot(alphas, ring_intensity+planet_intensity, label = 'Ring+Planet')

plt.xlabel(r'Phase angle $\alpha$')
plt.ylabel(r'Intensity (arbitrary)')
plt.legend()

plt.tight_layout()


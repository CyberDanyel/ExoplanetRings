import numpy as np
import exoring_functions
import scattering
import scipy.integrate as spi
import materials


# coordinate systems defined such that the observer is always along the x-axis
# planet always at origin
# phase angle alpha is aligned with the phi of the spherical coordinate system; the star is always 'equatorial'

# everything also assumes circular orbits


class Planet:
    def __init__(self, sc_law, radius, star):
        """
        Parameters
        ----------
            radius : float
        The radius of the sphere.
        """
        self.radius = radius
        self.sc_law = sc_law
        self.phase_curve = np.vectorize(self.phase_curve_single)  # vectorizing so that arrays of phase angles can be
        # input more easily than with a Python for loop
        self.star = star

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
        return np.sin(theta) * mu * mu_star * self.sc_law(alpha) * self.secondary_eclipse(theta, phi,
                                                                                          alpha)  # * (np.abs(
        # alpha) >= np.arcsin((self.star.radius + self.radius) / self.star.distance))
        # Only check secondary eclipse for certain alphas when you are close to the star for speed

    def phase_curve_single(self, alpha: float) -> float:
        """
        The unvectorized phase curve for an atmospheric scattering law independent of the angle of incidence
        Parameters
        ----------
            alpha : phase angle
        Returns
        -------
        The phase curve evaluated at the phase angle alpha
        """

        return self.sc_law(alpha) * self.secondary_eclipse_single(alpha) * scattering.lambert_phase_func(alpha)

    def phase_curve_multiple(self, alpha: float) -> float:
        """
        The unvectorized phase curve for an atmospheric scattering law dependent on the angle of incidence
        Parameters
        ----------
            alpha : phase angle
        Returns
        -------
        The phase curve evaluated at the phase angle alpha
        """
        theta_bounds = [0, np.pi]
        phi_bounds = [max(alpha - np.pi / 2, -np.pi / 2), min(alpha + np.pi / 2, np.pi / 2)]
        return exoring_functions.integrate2d(lambda theta, phi: self.phase_curve_integrand(theta, phi, alpha),
                                             bounds=[theta_bounds, phi_bounds], sigma=1e-3)

    def shadow_integrand(self, theta, alpha):
        '''
        The integrand for finding the amount of flux blocked by the secondary eclipse.
        This is for a 1D integral across theta.

        Parameters
        ----------
            theta : (float)
        The angle theta in a spherical coordinate system
            alpha : (float)
        The phase angle between the star, planet and observer

        Returns
        -------
        The flux density (in theta) of the area being blocked by the secondary eclipse
        '''
        R = self.star.distance
        r_star = self.star.radius
        r = self.radius
        offset = R * np.sin(alpha)
        h = np.sqrt(r_star ** 2 - r ** 2 * np.cos(theta) ** 2)

        if np.abs(offset + h) < np.abs(r * np.sin(theta)):
            phi_upper = np.arcsin((offset + h) / (r * np.sin(theta)))
        else:
            phi_upper = min(np.pi / 2, np.pi / 2 + alpha)

        if np.abs(offset - h) < np.abs(r * np.sin(theta)):
            phi_lower = np.arcsin((offset - h) / (r * np.sin(theta)))
        else:
            phi_lower = max(-np.pi / 2, alpha - np.pi / 2)

        term = lambda phi: 0.25 * (2 * phi * np.cos(alpha) - np.sin(alpha - 2 * phi)) * np.sin(theta) ** 3
        term_upper = term(phi_upper)
        term_lower = term(phi_lower)
        return term_upper - term_lower

    def secondary_eclipse_single(self, alpha: float) -> float:
        '''
        Calculates the fraction of flux blocked due to the secondary
        eclipse at some phase angle, assuming a Lambertian surface.
        Can be used in phase curve calculations for atmospheres with
        scattering laws independent of the incident direction of radiation

        Parameters
        ----------
            alpha : (float)
        The phase angle between star, planet and observer
        
        Returns
        -------
        The fraction of flux blocked at a phase angle 

        '''
        R = self.star.distance
        r_star = self.star.radius
        r = self.radius
        offset = R * np.sin(alpha)

        if np.abs(R * np.sin(alpha)) > (r + r_star) or np.cos(alpha) < 0:
            # no occlusion
            return 1.

        elif np.abs(offset) + r < r_star:
            # total occlusion
            return 0.

        else:
            # partial occlusion
            angle = min(np.pi / 2, np.arccos((r ** 2 + offset ** 2 - r_star ** 2) / (2 * r * np.abs(offset))))
            theta_upper = np.pi / 2 + angle
            theta_lower = np.pi / 2 - angle
            blocked = \
            spi.quad(np.vectorize(lambda theta: self.shadow_integrand(theta, alpha)), theta_lower, theta_upper,
                     epsabs=1e-3)[0]
            flux = scattering.lambert_phase_func(alpha)
            return (flux - blocked) / flux

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
    def __init__(self, sc_law, inner_rad, outer_rad, normal, star):
        self.inner_radius = inner_rad
        self.outer_radius = outer_rad
        self.sc_law = sc_law
        normal = np.array(normal)
        self.normal = normal / np.sqrt(np.sum(normal ** 2))
        self.secondary_eclipse = np.vectorize(self.analytic_secondary_eclipse)
        self.star = star

    def get_mu_star(self, alpha):
        """mu_star = cos(angle between star and normal to ring)"""
        star_pos = np.array([np.cos(alpha), np.sin(alpha), np.zeros(np.shape(alpha))])
        return np.dot(self.normal, star_pos)

    def get_mu(self):
        """mu = cos(angle between observer and normal to ring)"""
        obs_pos = np.array([1, 0, 0])
        return np.dot(self.normal, obs_pos)

    def phase_curve(self, alpha):
        """phase curve innit"""
        mu = self.get_mu()
        mu_star = self.get_mu_star(alpha)
        return mu * mu_star * self.sc_law(alpha) * (mu_star > 0) * self.secondary_eclipse(alpha)  # boolean prevents
        # forwards scattering

    def analytic_secondary_eclipse(self, alpha):
        '''
        A semi-analytical calculation (1D numerical calculations) for finding the 
        fractional amount of flux blocked by the secondary eclipse

        Parameters
        ----------
        alpha : phase angle

        Returns
        -------
        fractional value of the flux blocked due to the secondary eclipse

        '''
        if np.abs(alpha) > 2.1 * self.star.radius / self.star.distance:
            return 1.

        mu = self.get_mu()
        n_x, n_y, n_z = self.normal

        y_star = self.star.distance * np.sin(alpha)

        sin_theta = np.sqrt(1 - mu ** 2)
        cos_phi = n_z / sin_theta
        sin_phi = -n_y / sin_theta

        outer_area = exoring_functions.overlap_area(self.star.radius, self.outer_radius, mu, cos_phi, sin_phi, y_star)
        inner_area = exoring_functions.overlap_area(self.star.radius, self.inner_radius, mu, cos_phi, sin_phi, y_star)
        area_on_ring = outer_area - inner_area
        total_ring_area = mu * np.pi * (self.outer_radius ** 2 - self.inner_radius ** 2)  # - self.inner_radius**2)
        if area_on_ring < 0:
            print('Alpha: %.3f Area on outer: %.4f, area on inner: %.4f' % (alpha, outer_area, inner_area))
        return 1. - (area_on_ring / total_ring_area)

    def light_curve(self, alpha):
        return (self.outer_radius ** 2 - self.inner_radius ** 2) * self.phase_curve(alpha) * self.star.luminosity / (
                4 * self.star.distance ** 2)


class RingedPlanet(Planet):
    def __init__(self, planet_sc, planet_r, ring_sc, ring_inner_r, ring_outer_r, ring_normal, star):
        self.ring = Ring(ring_sc, ring_inner_r, ring_outer_r, ring_normal, star)
        Planet.__init__(self, planet_sc, planet_r, star)

    def light_curve(self, alpha):
        planet_light_curve = Planet.light_curve(self, alpha)
        ring_light_curve = Ring.light_curve(self.ring, alpha)
        return planet_light_curve + ring_light_curve


class ExoJupiter(Planet):
    def __init__(self, single_albedo, radius, star):
        sc_law = scattering.Jupiter(single_albedo)
        Planet.__init__(self, sc_law, radius, star)


s = 5.67037e-8  # stefan boltzmann constant


class Star:
    def __init__(self, luminosity, radius, distance, mass, planet=None):
        self.luminosity = luminosity
        self.radius = radius
        self.distance = distance
        self.mass = mass
        self.planet = planet
        self.T = (self.luminosity / (s * 4 * np.pi * radius)) ** 0.25

    def transit_function(self, alpha):
        # if abs(alpha) <= np.pi / 2:
        #    separation = self.distance * np.sin(abs(alpha)) For these angles the planet could never transit the sun
        if abs(alpha) >= np.pi / 2:
            separation = self.distance * np.sin(np.pi - abs(alpha))
        else:
            return self.luminosity
        if alpha < 0:
            separation_coord = -separation
        elif alpha > 0:
            separation_coord = separation
        else:
            print('problem with alpha')
            exit()
        if separation >= self.planet.radius + self.radius:  # planet does not occlude star
            return self.luminosity
        if separation < self.radius - self.planet.radius:  # planet fully in front of star
            occluded_frac = (self.planet.radius / self.radius) ** 2
            return (1 - occluded_frac) * self.luminosity
        x_0 = (self.planet.radius ** 2 - self.radius ** 2 + separation_coord ** 2) / (2 * separation_coord)
        if alpha > 0:
            integration = abs(
                exoring_functions.circle_section_integral(self.planet.radius, [x_0, self.planet.radius])) + abs(
                exoring_functions.circle_section_integral(self.radius,
                                                          [x_0 - separation_coord, -self.radius]))
        elif alpha < 0:
            integration = abs(
                exoring_functions.circle_section_integral(self.planet.radius, [x_0, -self.planet.radius])) + abs(
                exoring_functions.circle_section_integral(self.radius,
                                                          [x_0 - separation_coord, self.radius]))
        else:
            print('problem with alpha 2')
            exit()
        occluded_frac = integration / (np.pi * self.radius ** 2)  # Occluded fraction
        if occluded_frac > 1:
            print('error, occluded_frac > 1')
            exit()
        return (1 - occluded_frac) * self.luminosity

    def planck_function(self, wavelength):
        c = 3e8
        h = 6.626e-34
        k = 1.380649e-23
        return (2 * h * c ** 2 / wavelength ** 5) / np.exp((h * c / (k * wavelength * self.T)) - 1)

    def L(self, bandpass):
        'luminosity in a certain wavelength region'
        return spi.quad(self.planck_function, bandpass[0], bandpass[1])

    def light_curve(self, alphas):
        light_curve = [self.transit_function(alpha) for alpha in list(alphas)]
        return light_curve

import numpy as np
import exoring_functions


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

    def phase_curve_unvectorized(self, alpha: float) -> float:
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
        self.secondary_eclipse = np.vectorize(self.unvectorized_secondary_eclipse)
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

    def light_curve(self, alpha):
        return (self.outer_radius ** 2 - self.inner_radius ** 2) * self.phase_curve(alpha) * self.star.luminosity / (
                4 * self.star.distance ** 2)


class Star:
    def __init__(self, luminosity, radius, distance, mass, planet=None):
        self.luminosity = luminosity
        self.radius = radius
        self.distance = distance
        self.mass = mass
        self.planet = planet

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

    def light_curve(self, alphas):
        light_curve = [self.transit_function(alpha) for alpha in list(alphas)]
        return light_curve

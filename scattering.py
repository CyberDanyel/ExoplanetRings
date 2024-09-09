import numpy as np
import scipy.special as spe
import scipy.interpolate as spip
import scipy.integrate as spi
import PyMieScatt as msc
import materials

def lambert_phase_func(alpha):
    """The planetary (not single scattering) phase curve for Lambertian scattering"""
    return (2 / (3*np.pi)) * ((np.pi - np.abs(alpha)) * np.cos(np.abs(alpha)) + np.sin(np.abs(alpha)))


class SingleScatteringLaw:
    def __init__(self, albedo, func):
        '''
        The basic scattering law object - referred to as capital Phi in equations

        Parameters
        ----------
            albedo: (float) value between 0 and 1
            func: (callable) the single scattering phase function
        '''
        self.albedo = albedo
        self.func = func
        self.norm = 0.5 * spi.quadrature(lambda angle: np.sin(angle) * func(angle), 0, np.pi, maxiter = 1000)[0]
        # normalizing the phase function

    def __call__(self, alpha):
        return (self.albedo / self.norm) * self.func(np.pi - np.abs(alpha))


class Lambert(SingleScatteringLaw):
    def __init__(self, albedo):
        '''
        An object implementing Lambertian scattering

        Parameters
        ----------
            albedo: (float) value between 0 and 1
        '''
        SingleScatteringLaw.__init__(self, albedo, lambda x: 1.)


class Rayleigh(SingleScatteringLaw):
    def __init__(self, albedo):
        '''
        An object implementing Rayleigh scattering

        Parameters
        ----------
            albedo: (float) value between 0 and 1
        '''
        SingleScatteringLaw.__init__(self, albedo, Rayleigh.rayleigh_func)

    def rayleigh_func(theta):
        return (1 + np.cos(theta) ** 2)


class HG(SingleScatteringLaw):
    # henyey-greenstein
    def __init__(self, g, albedo):
        '''
        An object implementing the Henyey-Greenstein phase function as a single scattering law
        Parameters
        ----------
            g: (float) value between -1 and 1 representing the forward-backward asymmetry of the phase function
            albedo: (float) value between 0 and 1
        '''
        self.g = g
        SingleScatteringLaw.__init__(self, albedo, self.hg_func)

    def hg_func(self, theta):
        'the Henyey-Greenstein phase function'
        return (2 * (1 - self.g ** 2)) / (1 + self.g ** 2 - 2 * self.g * np.cos(theta)) ** (1.5)


class SingleEmpirical(SingleScatteringLaw):
    def __init__(self, filename, albedo):
        """
        A single scattering law for phase functions derived from empirical data
        Parameters
        ----------
        filename: (str) name of the file containing the data
        albedo: (float) a value between 0 and 1
        """
        self.filename = filename
        self.points = np.loadtxt(filename, delimiter=',')
        emp_func = spip.CubicSpline(self.points[0], self.points[1])
        SingleScatteringLaw.__init__(self, albedo, emp_func)


class Mie(SingleScatteringLaw):
    def __init__(self, albedo, X, m):
        """
        An object implementing Mie scattering for a single particle size and incident wavelength

        Parameters
        ----------
            albedo: (float) a value between 0 and 1
            X: (float) the Mie size parameter (2pi*radius/wavelength)
            m: (complex) the complex refractive index of the particle material
        """
        self.X = X
        self.m = m
        SingleScatteringLaw.__init__(self, albedo, np.vectorize(self.mie_func))

    def mie_func(self, theta):
        """The Mie scattering phase function"""
        S1, S2 = msc.MieS1S2(self.m, self.X, np.cos(theta))
        return np.abs(S1) ** 2 + np.abs(S2) ** 2

class Jupiter(SingleScatteringLaw):
    # from Dyudina et al. 2005
    def __init__(self, albedo):
        """
        An empirical phase function object for Jupiter's atmosphere, based on Dyudina et al. 2005

        Parameters
        ----------
        albedo: (float) a value between 0 and 1
        """
        self.g1 = 0.8
        self.g2 = -.38
        self.f = 0.9
        SingleScatteringLaw.__init__(self, albedo, self.jupiter_func)

    def hg_func(self, g, theta):
        return (2 * (1 - g ** 2)) / (1 + g ** 2 - 2 * g * np.cos(theta)) ** (1.5)

    def jupiter_func(self, theta):
        return self.f * self.hg_func(self.g1, theta) + (1 - self.f) * self.hg_func(self.g2, theta)


class RandJ(Rayleigh, Jupiter):
    def __init__(self, albedo, ray_to_jup_ratio = (1,1)):
        self.ray_to_jup_ratio = ray_to_jup_ratio
        Rayleigh.__init__(self, albedo)
        Jupiter.__init__(self, albedo)
        SingleScatteringLaw.__init__(self, albedo, self.randj_func)

    def randj_func(self, theta):
        return self.ray_to_jup_ratio[0]*self.rayleigh_func(theta) + self.ray_to_jup_ratio[1]*self.jupiter_func(theta)

class WavelengthDependentScattering(Jupiter, Rayleigh, Lambert, Mie, SingleEmpirical):
    def __init__(self, material, bandpass, inc_spec):
        """
        A wavelength dependent single scattering phase function.
        The phase function at each wavelength is taken from the material object.
        The average scattering function is averaged across the bandpass - weighted by the incident spectrum

        Parameters
        ----------
        material: (materials.Atmosphere or materials.RingMaterial) the material for which to create a scattering object
        bandpass: (float, float) a tuple of wavelengths representing a rectangular bandpass
        inc_spec: (function) a function representing the spectrum of the light incident on the material
        """
        self.material = material
        self.bandpass = bandpass
        self.inc_spec = inc_spec  # spectrum of the incident light for weighting scattering functions
        self.spec_norm = spi.quad(inc_spec, bandpass[0], bandpass[1], limit=100)[0]
        albedo = spi.quad(lambda wav: material.albedo(wav) * self.wavelength_weighting(wav), bandpass[0], bandpass[1], limit=1000)[0]
        if isinstance(material, materials.RingMaterial):
            angles = np.linspace(0, np.pi, 1000)
            vals = []
            for angle in angles:
                # int_val = spi.quad(lambda lam:material.phase_func(angle, lam), bandpass[0], bandpass[1])[0]
                # this integral is incredibly slow, the following code is a simpler but quicker Riemann sum
                lams = material.wavelengths
                lams = lams[(lams >= bandpass[0]) * (lams <= bandpass[1])]
                dlams = (lams - np.roll(lams, 1))[1:]
                integrand = []
                for i, dlam in enumerate(dlams):
                    integrand.append(
                        material.phase_funcs[lams[i]](angle) * dlam * self.wavelength_weighting(lams[i]) * material.albedo(lams[i]))
                int_val = np.sum(integrand)
                avg_val = int_val
                vals.append(avg_val)
            func = spip.CubicSpline(angles, vals)
            SingleScatteringLaw.__init__(self, albedo, func)
        elif isinstance(material, materials.Atmosphere):
            material.sc_class.__init__(self, albedo)
        else:
            material.sc_class.__init__(self, albedo)
        

    def wavelength_weighting(self, wavelength):
        return self.inc_spec(wavelength) / self.spec_norm

import numpy as np
import scipy.special as spe
import scipy.interpolate as spip
import scipy.integrate as spi
import PyMieScatt as msc

def lambert_phase_func(alpha):
    #important enough to make its own thing
    return 2/3 * ((np.pi-np.abs(alpha))*np.cos(np.abs(alpha))+np.sin(np.abs(alpha)))
    

class SingleScatteringLaw:
    def __init__(self, albedo, func):
        self.albedo = albedo
        self.func = func
        self.norm = 2*np.pi*spi.quad(lambda angle:np.sin(angle)*func(angle), 0, np.pi, limit=1080)[0]
    def __call__(self, alpha):
        return (self.albedo/self.norm) * self.func(np.pi-np.abs(alpha))

class Lambert(SingleScatteringLaw):
    def __init__(self, albedo):
        SingleScatteringLaw.__init__(self, albedo, lambda x:1.)
        
class Rayleigh(SingleScatteringLaw):
    def __init__(self, albedo):
        SingleScatteringLaw.__init__(self, albedo, self.rayleigh_func)
        
    def rayleigh_func(self, theta):
        return (1+np.cos(theta)**2)
       
class HG(SingleScatteringLaw):
    #henyey-greenstein
    def __init__(self, g, albedo):
        self.g = g
        SingleScatteringLaw.__init__(self, albedo, self.hg_func)
    
    def hg_func(self, theta):
        return (2*(1 - self.g**2)) / (1 + self.g**2 - 2*self.g*np.cos(theta))**(1.5)
        
class SingleEmpirical(SingleScatteringLaw):
    def __init__(self, filename, albedo):
        self.filename = filename
        self.points = np.loadtxt(filename, delimiter = ',')
        emp_func = spip.CubicSpline(self.points[0], self.points[1])
        SingleScatteringLaw.__init__(self, albedo, emp_func)

class Mie(SingleScatteringLaw):
    def __init__(self, albedo, X, m):
        self.X = X
        self.m = m
        SingleScatteringLaw.__init__(self, albedo, np.vectorize(self.mie_func))
        
    def mie_func(self, theta):
        S1, S2 = msc.MieS1S2(self.m, self.X, np.cos(theta))
        return np.abs(S1)**2 + np.abs(S2)**2

#general functions independent of specific scattering situation
def psi(x, n):
    return x*spe.spherical_jn(n, x)
def zeta(x, n):
    return np.sqrt((np.pi * x)/2) * spe.hankel2(n+0.5, x)
def psi_prime(x, n):
    dx = 1e-8
    return (psi(x+dx, n) - psi(x, n))/dx
def zeta_prime(x, n):
    dx = 1e-8
    return (zeta(x+dx, n) - zeta(x, n))/dx


class Jupiter(SingleScatteringLaw):
    #from Dyudina et al. 2005
    def __init__(self, albedo):
        self.g1 = 0.8
        self.g2 = -.38
        self.f = 0.9
        SingleScatteringLaw.__init__(self, albedo, self.jupiter_func)
    
    def hg_func(self, g, theta):
        return (2*(1 - g**2)) / (1 + g**2 - 2*g*np.cos(theta))**(1.5)
        
    def jupiter_func(self, theta):
        return self.f*self.hg_func(self.g1, theta) + (1-self.f)*self.hg_func(self.g2, theta)
    
class WavelengthDependentScattering(SingleScatteringLaw):
    def __init__(self, material, bandpass, inc_spec):
        self.material = material
        self.bandpass = bandpass
        self.bandwidth = bandpass[1] - bandpass[0]
        self.inc_spec = inc_spec #spectrum of the incident light for weighting scattering functions
        self.spec_norm = spi.quad(inc_spec, bandpass[0], bandpass[1], limit = 100)[0]
        albedo = spi.quad(lambda wav:material.albedo(wav)*self.wavelength_weighting(wav), bandpass[0], bandpass[1], limit=1000)[0]
        # we dont need to normalize albedo by bandpass since wavelength_weights is already normalized by bandpass
        angles = np.linspace(0, np.pi, 1000)
        vals = []
        for angle in angles:
            #int_val = spi.quad(lambda lam:material.phase_func(angle, lam), bandpass[0], bandpass[1])[0]
            #this integral is incredibly slow, the following code is a simpler but quicker Riemann sum
            lams = material.wavelengths
            lams = lams[(lams >= bandpass[0]) * (lams <= bandpass[1])]
            dlams = (lams - np.roll(lams, 1))[1:]
            integrand = []
            for i, dlam in enumerate(dlams):
               integrand.append(material.phase_funcs[lams[i]](angle)*dlam*self.wavelength_weighting(lams[i])*material.albedo(lams[i]))
            int_val = np.sum(integrand)
            avg_val = int_val / self.bandwidth
            vals.append(avg_val)
        func = spip.CubicSpline(angles, vals)
        SingleScatteringLaw.__init__(self, albedo, func)
    
    def wavelength_weighting(self, wavelength):
        return self.inc_spec(wavelength)/self.spec_norm
        

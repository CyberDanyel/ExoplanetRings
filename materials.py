import numpy as np
import pandas as pd
import scipy.interpolate as spint
import scipy.integrate as spi
import platon
from platon.TP_profile import Profile
from platon.eclipse_depth_calculator import EclipseDepthCalculator


class _MatPhaseFuncs:
    'phase function interpolated for scattering angle but not wavelength'
    def __init__(self, data):
        self.data = data
        phase_funcs = []
        wavelengths = []
        for wavelength in self.data:
            angles = self.data.index
            phase_data = self.data[wavelength]
            phase_funcs.append(spint.PchipInterpolator(angles, phase_data))
            wavelengths.append(wavelength)
        self.wavelengths = np.array(wavelengths)
        self.df = pd.DataFrame([phase_funcs], columns = wavelengths)
        
    def __getitem__(self, i):
        return self.df[i][0]
        
    def __len__(self):
        return len(self.wavelengths)
        

class RingMaterial:
    def __init__(self, filename, nang, nlam):
        '''
        A material for a ring, using a single scattering approximation.
        The single scattering cross section is used to calculate the
        wavelength dependent albedo, and the rest of the radiation is absorbed
        (no forward scattering)
        
        Parameters
        -----------
            filename: including the file path if necessary. This should be formatted
        as an optool output file using the -radmc option.
            nang: the number of points in the angular grid used by optool
            nlam: the number of points in the wavelength grid used by optool
        '''
        opacity_data = np.loadtxt(filename, skiprows = 42, max_rows = nlam)
        wavelengths = opacity_data[:,0] * 1e-6
        self.k_sc = opacity_data[:,2] * 1e-4 # necessary for normalization, but the rest of the model uses SI
        self.k_ab = opacity_data[:,1] * 1e-4
        self.wavelengths = wavelengths
        self.albedos = self.k_sc/(self.k_sc + self.k_ab)
        self.albedo_func = spint.CubicSpline(self.wavelengths, self.albedos)
        
        angles = np.loadtxt(filename, skiprows = 42 + 1 + nlam, max_rows = nang) * (np.pi/180) # why is it not already in radians, ew
        sc_data = np.loadtxt(filename, skiprows = 42 + 1 + nlam + 1 + nang)
        
        sc_data = sc_data.reshape(nlam, nang, 6)[:,:,0].T # reshaping data so that each row is assigned an angle and each column is a wavelength
        sc_data *= 4*np.pi/np.broadcast_to(self.k_sc, sc_data.shape) # normalization - optool output is normalized to scattering cross section
        self.data = pd.DataFrame(sc_data, index = angles, columns = wavelengths)
        
        self.phase_funcs = _MatPhaseFuncs(self.data) # phase functions not interpolated in wavelength
        self.phase_interpol = spint.RegularGridInterpolator((angles, wavelengths), sc_data, method = 'pchip', fill_value = None) # fully 2D interpolated phase function
   
    def phase_func(self, angle, wavelength):
        '''
        Returns the phase function of the material at any wavelength

        Parameters
        ----------
            angle: Scattering angle (NOT phase angle).
            wavelength: wavelength of light.
        '''
        return self.phase_interpol((angle, wavelength))

    def albedo(self, wavelength):
        'The wavelength dependent albedo'
        return self.albedo_func(wavelength)
    
# possible to-do: class RingAggregate(RingMaterial)
# would use radmc and include forward scattering

s = 5.67037e-8  # stefan boltzmann constant

class Atmosphere:
    def __init__(self, platon_params, planet_params, star):
        P_1, alpha_1, alpha_2, P_3 = platon_params
        sc_class, planet_mass, planet_radius = planet_params
        T = 0.5*(star.luminosity/(np.pi*s))**0.25 * star.distance**-0.5
        self.sc_class = sc_class
        self.T = T
        self.p = Profile()
        self.p.set_parametric(T, P_1, alpha_1, alpha_2, P_3, T)
        self.calc = EclipseDepthCalculator()
        self.wavelengths, self.depths = self.calc.compute_depths(self.p, star.distance, planet_mass, planet_radius, star.T)
        self.albedos = {}
        for i, wavelength in enumerate(self.wavelengths):
            sc_law = sc_class(albedo=1)
            A_g = 2/3 * sc_law(0)
            depth_unitalbedo = planet_radius**2/(4*star.distance**2)#*A_g
            self.depth_unitalbedo = (depth_unitalbedo + np.pi*planet_radius**2 * s * T**4/star.luminosity)
            self.albedos[wavelength] = self.depths[i]/depth_unitalbedo
        self.albedo_func = spint.PchipInterpolator(self.wavelengths, list(self.albedos.values()))

    def albedo(self, wavelength):
        'The wavelength dependent albedo'
        return self.albedo_func(wavelength)
    

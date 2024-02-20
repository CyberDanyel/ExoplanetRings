import numpy as np
import pandas as pd
import scipy.interpolate as spint
import scipy.integrate as spi
import platon
from platon.TP_profile import Profile
from platon.transit_depth_calculator import TransitDepthCalculator

R_JUP = 69.911e9
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
    def __init__(self, sc_class, planet_params, star, meters_per_length_unit=1, invert=False):
        planet_mass, planet_radius = planet_params[:2]
        planet_radius *= meters_per_length_unit
        star.radius *= meters_per_length_unit
        star.luminosity *= meters_per_length_unit**2
        star.distance *= meters_per_length_unit
        self.meters_per_length_unit = meters_per_length_unit
        T = 0.5*(star.luminosity/(np.pi*s))**0.25 * star.distance**-0.5 # result independent of length unit
        self.sc_class = sc_class
        self.T = T
        self.calc = TransitDepthCalculator()
        if invert:
            self.wavelengths, self.depths, info_dict = self.calc.compute_depths(star.radius, planet_mass, planet_radius, self.T, CO_ratio=0.425381, logZ=-1,  add_scattering=True, stellar_blackbody=True, full_output=True)
            self.wavelengths *= R_JUP
            self.albedos = np.zeros(np.shape(self.wavelengths))
            depth_unitalbedo = (max(info_dict['radii'])**2 - min(info_dict['radii'])**2)/star.radius**2
            planetless_depths = self.depths - min(info_dict['radii'])**2/star.radius**2
            self.depth_unitalbedo = depth_unitalbedo
            self.albedos += planetless_depths/depth_unitalbedo
        else:
            self.wavelengths, self.depths, info_dict = self.calc.compute_depths(star.radius, planet_mass, planet_radius, self.T, CO_ratio=0.425381, logZ=-1,  add_scattering=False, stellar_blackbody=True, full_output=True)
            self.albedos = np.zeros(np.shape(self.wavelengths))
            depth_unitalbedo = (max(info_dict['radii'])**2 - min(info_dict['radii'])**2)/star.radius**2
            depth_unitalbedo += np.pi*planet_radius**2 * s * T**4/star.luminosity
            self.depth_unitalbedo = depth_unitalbedo
            for i, wavelength in enumerate(self.wavelengths):
                self.albedos[i] += ((max(info_dict['radii'])**2/star.radius**2) - self.depths[i])/depth_unitalbedo
        self.albedo_func = spint.PchipInterpolator(self.wavelengths, self.albedos)
        star.radius /= meters_per_length_unit
        star.luminosity /= meters_per_length_unit**2
        star.distance /= meters_per_length_unit
    def albedo(self, wavelength):
        'The wavelength dependent albedo'
        return self.albedo_func(wavelength)


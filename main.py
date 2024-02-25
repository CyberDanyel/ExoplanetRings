if __name__ == '__main__':
    import json
    import exoring_objects
    import scattering
    import numpy as np
    import json
    import time
    import matplotlib.pyplot as plt
    import exoring_functions
    from matplotlib.ticker import FuncFormatter
    import matplotlib.ticker as tck
    import materials
    import pickle
    import matplotlib

    with open('constants.json') as json_file:
        constants = json.load(json_file)

    bandpass = (10e-6, 14e-6)

    start = time.time()

    star = exoring_objects.Star(5800, constants['R_SUN_TO_R_JUP'], .1*constants['AU_TO_R_JUP'], 1.)
    # star_obj = exoring_objects.Star(1, SUN_TO_JUP, 0.5 * AU_TO_JUP, 1)

    scattering_laws = dict()
    atmos = materials.Atmosphere(scattering.Jupiter, [constants['M_JUP'], 1], star, meters_per_length_unit = constants['R_JUP'])
    ice = materials.RingMaterial('materials/saturn_small_ice.inp', 361, 500)
    silicate = materials.RingMaterial('materials/silicate_small.inp', 361, 500)

    scattering_laws['ice'] = scattering.WavelengthDependentScattering(ice, bandpass, star.planck_function)
    scattering_laws['silicates'] = scattering.WavelengthDependentScattering(silicate, bandpass, star.planck_function)
    scattering_laws['atmosphere'] = scattering.WavelengthDependentScattering(atmos, bandpass, star.planck_function)
    scattering_laws['mie'] = scattering.Mie(1., 100e-6 / np.mean(bandpass), 1.5 + .1j)
    scattering_laws['rayleigh'] = scattering.Rayleigh(0.1)
    scattering_laws['jupiter'] = scattering.Jupiter(1)

    with open('scattering_laws.pkl', 'wb') as f:
        pickle.dump(scattering_laws, f)

    import fitting
    model_parameters = {'radius': 1,
                        'disk_gap': 1, 'ring_width': 1,
                        'ring_normal': np.array([1, 1, 0])}

    test_ring_planet = exoring_objects.RingedPlanet(scattering_laws['atmosphere'], model_parameters['radius'], scattering_laws['silicates'],
                                                    model_parameters['radius'] + model_parameters['disk_gap'],
                                                    model_parameters['radius'] + model_parameters['disk_gap'] +
                                                    model_parameters['ring_width'],
                                                    model_parameters['ring_normal'], star)

    ring_data = fitting.generate_data(test_ring_planet)

    Data = fitting.DataObject(ring_data, star)
    #print(str(Data.fit_data_ring(sc_planet, sc_sil, init_guess)))
    Data.produce_corner_plot(model_parameters, {'radius': (0, 1.1), 'disk_gap': (0, 2.5), 'ring_width': (0.5, 1.3)}, 70,
                             planet_sc_law='jupiter', ring_sc_law='rayleigh', ringed=True, log=False, multiprocessing=True)
    #Data.run_ringed_model('atmosphere', 'silicates', model_parameters)
    # Data.disperse_models(test_ring_planet, scattering.Jupiter, scattering.Rayleigh, ('ring_width', 'radius'), model_parameters)
    # param_sets1 = Data.create_various_model_parameters(radius = (0.09, 0.83, 1), ring_width = (JUP_SATURN_LIKE_RING, 10, (1/3)*JUP_HILL_RAD), disk_gap = (0.01))
    # param_sets2 = Data.create_various_model_parameters(radius = (0.09, 0.83, 1), disk_gap = (JUP_SATURN_LIKE_RING, 10, (1/3)*JUP_HILL_RAD), ring_width = (1))
    # param_sets3 = Data.create_various_model_parameters(ring_width=(JUP_SATURN_LIKE_RING, 10, 50),
    # disk_gap=(JUP_SATURN_LIKE_RING, 10, 50),
    # radius=(1))
    # Data.run_many_ringless_models(scattering.Jupiter, param_sets)
    # Data.run_many_ringed_models(scattering.Jupiter, scattering.Rayleigh, param_sets2, changing_parms=('radius','disk_gap'), static_param=('W', 1))
    # Data.run_many_ringed_models(scattering.Jupiter, scattering.Rayleigh, param_sets3, changing_parms=('ring_width','disk_gap'), static_param=('R', 1))
    # Data.run_many_ringed_models(scattering.Jupiter, scattering.Rayleigh, [model_parameters for i in range(9)])
    # Data.range_search_fitting(search_ranges_planet, 0.5, bounds_planet, [scattering.Lambert, scattering.Jupiter])
    # Data.range_search_fitting(search_ranges_ring, 0.5, bounds_ring, [scattering.Lambert, scattering.Jupiter], [scattering.Rayleigh])
    # Data.plot_best_ringfit()
    # Data.plot_best_planetfit()
    end = time.time()
    print('Time taken for code execution:', round(end - start, 2), 'seconds')

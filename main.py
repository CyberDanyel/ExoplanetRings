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

    meters_per_length_unit = constants['R_JUP']
    star = exoring_objects.Star(5800, constants['R_SUN_TO_R_JUP'], .1 * constants['AU_TO_R_JUP'], 1.)
    # star_obj = exoring_objects.Star(1, SUN_TO_JUP, 0.5 * AU_TO_JUP, 1)

    scattering_laws = dict()
    atmos = materials.Atmosphere(scattering.Jupiter, [constants['M_JUP'], 1], star,
                                 meters_per_length_unit=meters_per_length_unit)
    silicate = materials.RingMaterial('materials/silicate_small.inp', 361, 500)

    scattering_laws['silicates'] = scattering.WavelengthDependentScattering(silicate, bandpass, star.planck_function)
    scattering_laws['atmosphere'] = scattering.WavelengthDependentScattering(atmos, bandpass, star.planck_function)
    scattering_laws['rayleigh'] = scattering.Rayleigh(0.1)
    scattering_laws['jupiter'] = scattering.Jupiter(1)

    with open('scattering_laws.pkl', 'wb') as f:
        pickle.dump(scattering_laws, f)

    import fitting

    with open('scattering_laws.pkl', 'rb') as f:
        scattering_laws = pickle.load(f)

    theta, phi = np.pi / 2, np.pi / 4
    model_parameters = {'radius': 1,
                        'disk_gap': 1, 'ring_width': 1,
                        'theta': theta,
                        'phi': phi
                        }

    test_ringless_planet = fitting.FittingPlanet(scattering_laws['atmosphere'], star, model_parameters)
    test_ring_planet = fitting.FittingRingedPlanet(scattering_laws['atmosphere'], scattering_laws['silicates'], star, model_parameters)

    ringless_data = fitting.generate_data(test_ringless_planet)
    ring_data = fitting.generate_data(test_ring_planet)


    #Data = fitting.DataObject(ringless_data, star)
    #planet_init_guess = {'radius': 1.4}
    #Data.fit_data_planet('atmosphere', planet_init_guess)

    Data = fitting.DataObject(ring_data, star)
    #Data.display_ringed_model('atmosphere', 'silicates', model_parameters)
    #planet_init_guess = {'radius': 1}
    #ring_init_guess = {'radius': 1, 'disk_gap': 1, 'ring_width': 1, 'theta':np.pi/2, 'phi':np.pi/4}
    #planet_bounds = {'radius': (0.1,10)}
    #ring_bounds = {'radius': (0.1,2), 'disk_gap': (0.1,2), 'ring_width': (0.5,1.5), 'theta':(0, np.pi), 'phi':(-np.pi/2, np.pi/2)}
    #minimiser = Data.fit_data_planet('atmosphere', planet_init_guess, planet_bounds)
    #print(minimiser)
    #minimiser = Data.fit_data_ring('atmosphere', 'silicates', ring_init_guess, ring_bounds)
    #print(str(Data.fit_data_ring(sc_planet, sc_sil, init_guess)))
    Data.produce_corner_plot(model_parameters,
                             {'radius': (0.7, 1.3, 70), 'disk_gap': (0, 3, 70), 'ring_width': (2/5, 8/5, 70)},
                             planet_sc_law='atmosphere', ring_sc_law='silicates', ringed=True, log=False,
                             multiprocessing=True, save_data=False)
    # , 'theta': (0, np.pi / 2, 2), phi': (-np.pi / 2, np.pi / 2, 3)
    # Data.run_ringed_model('atmosphere', 'silicates', model_parameters)
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
    #minimiser.migrad()
    #print(minimiser)
    #fig, ax = minimiser.draw_mnmatrix()
    #fig.savefig('images/hey', dpi=600)
    end = time.time()
    print('Time taken for code execution:', round(end - start, 2), 'seconds')

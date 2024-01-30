if __name__ == '__main__':
    import exoring_objects
    import scattering
    import numpy as np
    import fitting
    import json
    import time
    import matplotlib.pyplot as plt
    import exoring_functions
    from matplotlib.ticker import FuncFormatter
    import matplotlib.ticker as tck

    start = time.time()

    AU = 1.495978707e13
    L_SUN = 3.828e33
    R_JUP = 6.9911e9
    R_SUN = 6.957e10
    JUP_TO_AU = AU / R_JUP
    SUN_TO_JUP = R_SUN / R_JUP
    SUN_TO_AU = AU / R_SUN
    AU_TO_JUP = AU / R_JUP
    JUP_HILL_RAD = 0.355*JUP_TO_AU
    JUP_SATURN_LIKE_RING = 1.144

    star = exoring_objects.Star(1, SUN_TO_JUP, 0.1 * JUP_TO_AU, 1)
    '''
    planet = exoring_objects.Planet(scattering.Jupiter(1), 1, star)
    
    star.planet = planet
    
    ring_normal = np.array([1., 1., .1])
    ring_normal /= np.sqrt(np.sum(ring_normal ** 2))
    ring_law = scattering.Mie(1, 4, 1.5)
    ring = exoring_objects.Ring(ring_law, 1.1, 2., ring_normal, star)
    
    alphas = np.array(
        list(np.linspace(-np.pi, -.06, 1000)) + list(np.linspace(-.06, .06, 3000)) + list(np.linspace(.06, np.pi, 1000)))
    
    planet_curve = planet.light_curve(alphas)
    ring_curve = ring.light_curve(alphas)
    #star_curve = star.light_curve(alphas)
    
    plt.style.use('the_usual.mplstyle')
    
    fig1, ax1 = exoring_functions.generate_plot_style()
    ax1.plot(alphas/np.pi, planet_curve, label='Planet')
    ax1.plot(alphas/np.pi, ring_curve, label='Ring')
    ax1.plot(alphas/np.pi, planet_curve + ring_curve, label='Ring + Planet')
    ax1.legend()
    plt.savefig('images/Planet & Rings Light Curve')
    
    #fig2, ax2 = exoring_functions.generate_plot_style()
    #ax2.plot(alphas/np.pi, planet_curve+ring_curve+star_curve, label='Star + Planet + Ring')
    #ax2.legend()
    #plt.savefig('images/Full Light Curve')
    '''
    '''
        search_ranges_ring = {'radius': (0.1, 1.9), 'disk_gap': (0, 3), 'ring_width': (1, 2),
                          'ring_normal': [(0.1, 1), (0.1, 1), (0, 0)],
                          'planet_sc_args': {'albedo': (0.1, 1)},
                          'ring_sc_args': {'albedo': (0.1, 1)}}
    bounds_ring = {'radius': (0, star.radius), 'disk_gap': (0, np.inf), 'ring_width': (0.1, np.inf),
                   'ring_normal': [(0, 1), (0, 1), (0, 1)],
                   'planet_sc_args': {'albedo': (0, 1)},
                   'ring_sc_args': {'albedo': (0, 1)}}
    search_ranges_planet = {'radius': (0.1, 1.9), 'planet_sc_args': {'albedo': (0.1, 1)}}
    '''

    star_obj = exoring_objects.Star(1, SUN_TO_JUP, 0.1 * AU_TO_JUP, 1)
    model_parameters = {'radius': 1,
                        'disk_gap': 1, 'ring_width': 1,
                        'ring_normal': np.array([1., 1., 0]),
                        'planet_sc_args': {'albedo': 1},
                        'ring_sc_args': {'albedo': 0.1}}
    test_ring_planet = exoring_objects.RingedPlanet(scattering.Jupiter(model_parameters['planet_sc_args']['albedo']), model_parameters['radius'], scattering.Rayleigh(model_parameters['ring_sc_args']['albedo']), model_parameters['radius']+model_parameters['disk_gap'], model_parameters['radius']+model_parameters['disk_gap']+model_parameters['ring_width'],
                                                    np.array([1., 1., 0]), star_obj)

    ring_data = fitting.generate_data(test_ring_planet)

    bounds_planet = {'radius': (0, star.radius), 'planet_sc_args': {'albedo': (0, 1)}}

    Data = fitting.Data_Object(ring_data, star_obj)
    Data.produce_corner_plot(model_parameters,{'radius':(0.9,1.1), 'disk_gap':(0,3), 'ring_width':(0.5,1.6)},ringed = True, number = 70, log = False, multiprocessing=True, planet_sc_law=scattering.Jupiter, ring_sc_law=scattering.Rayleigh)
    #Data.run_ringed_model(scattering.Jupiter, scattering.Rayleigh, model_parameters)
    #param_sets1 = Data.create_various_model_parameters(radius = (0.09, 0.83, 1), ring_width = (JUP_SATURN_LIKE_RING, 10, (1/3)*JUP_HILL_RAD), disk_gap = (0.01))
    #param_sets2 = Data.create_various_model_parameters(radius = (0.09, 0.83, 1), disk_gap = (JUP_SATURN_LIKE_RING, 10, (1/3)*JUP_HILL_RAD), ring_width = (1))
    #param_sets3 = Data.create_various_model_parameters(ring_width=(JUP_SATURN_LIKE_RING, 10, 50),
                                                       #disk_gap=(JUP_SATURN_LIKE_RING, 10, 50),
                                                       #radius=(1))
    #Data.run_many_ringless_models(scattering.Jupiter, param_sets)
    #Data.disperse_models(test_ring_planet, scattering.Jupiter, scattering.Rayleigh, ('disk_gap', 'ring_width'), model_parameters)
    #Data.run_many_ringed_models(scattering.Jupiter, scattering.Rayleigh, param_sets2, changing_parms=('radius','disk_gap'), static_param=('W', 1))
    #Data.run_many_ringed_models(scattering.Jupiter, scattering.Rayleigh, param_sets3, changing_parms=('ring_width','disk_gap'), static_param=('R', 1))
    #Data.run_many_ringed_models(scattering.Jupiter, scattering.Rayleigh, [model_parameters for i in range(9)])
    # Data.range_search_fitting(search_ranges_planet, 0.5, bounds_planet, [scattering.Lambert, scattering.Jupiter])
    # Data.range_search_fitting(search_ranges_ring, 0.5, bounds_ring, [scattering.Lambert, scattering.Jupiter], [scattering.Rayleigh])
    # Data.plot_best_ringfit()
    # Data.plot_best_planetfit()
    end = time.time()
    print('Time taken:', round(end - start,2), 'seconds')

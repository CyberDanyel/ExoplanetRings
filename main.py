if __name__ == '__main__':
    import exoring_objects
    import scattering
    import numpy as np
    import fitting
    import json
    import matplotlib.pyplot as plt
    import exoring_functions
    from matplotlib.ticker import FuncFormatter
    import matplotlib.ticker as tck

    AU = 1.495978707e13
    L_SUN = 3.828e33
    R_JUP = 6.9911e9
    R_SUN = 6.957e10
    JUP_TO_AU = AU / R_JUP
    SUN_TO_JUP = R_SUN / R_JUP
    SUN_TO_AU = AU / R_SUN
    AU_TO_JUP = AU / R_JUP

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

    star_obj = exoring_objects.Star(1, SUN_TO_JUP, 0.1 * AU_TO_JUP, 1)
    test_ring_planet = exoring_objects.RingedPlanet(scattering.Jupiter(1), 1, scattering.Rayleigh(0.9), 2, 3,
                                                    np.array([1., 1., 0]), star_obj)
    ring_data = fitting.generate_data(test_ring_planet)

    search_ranges_ring = {'radius': (0.1, 1.9), 'disk_gap': (0, 3), 'ring_width': (1, 2),
                          'ring_normal': [(0.1, 1), (0.1, 1), (0, 0)],
                          'planet_sc_args': {'albedo': (0.1, 1)},
                          'ring_sc_args': {'albedo': (0.1, 1)}}
    bounds_ring = {'radius': (0, np.inf), 'disk_gap': (0, np.inf), 'ring_width': (0.1, np.inf),
                   'ring_normal': [(0, 1), (0, 1), (0, 1)],
                   'planet_sc_args': {'albedo': (0, 1)},
                   'ring_sc_args': {'albedo': (0, 1)}}
    Fit = fitting.PerformFit(ring_data, star_obj)
    # result_planetfit = Fit.fit_data_planet(scattering.Jupiter, init_guess_planet)
    result = Fit.perform_fitting(search_ranges_ring, 0.5, bounds_ring, [scattering.Jupiter], [scattering.Rayleigh])
    print('result length is', len(result))
    print('result is', result)
    with open('data.json', 'w') as f:
        json.dump(result, f)

import matplotlib.pyplot as plt
import numpy as np
import exoring_objects
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

plt.style.use('the_usual.mplstyle')

star = exoring_objects.Star(1, 1 * SUN_TO_JUP, .1 * JUP_TO_AU, 1)

planet = exoring_objects.Planet(0.52, 1, star)

ring_normal = np.array([1., 1., 0.])
ring_normal /= np.sqrt(np.sum(ring_normal * ring_normal))

ring_normal2 = np.array([1., 0., 0.0])
ring_normal2 /= np.sqrt(np.sum(ring_normal * ring_normal))

ring = exoring_objects.Ring(0.7, 1, 2., ring_normal, star)
star.planet = planet  # Should use inheritance to prevent this being necessary
# ring2 = Ring(0.8, 1, 10, ring_normal2, star)

animation = exoring_functions.Animation(planet, star, ring, including_star=True)
animation.generate_animation()

plt.style.use('the_usual.mplstyle')
# plt.subplots_adjust(top=2.1, bottom=2, tight_layout=True)
fig, ax = plt.subplots()
ax.plot(animation.alphas / np.pi, animation.planet_curve, label='Planet')
ax.plot(animation.alphas / np.pi, animation.ring_curve, label='Ring')
ax.plot(animation.alphas / np.pi, animation.planet_curve + animation.ring_curve, label='Ring + Planet')
ax.xaxis.set_major_formatter(FuncFormatter(exoring_functions.format_fraction_with_pi))
ax.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
ax.set_xlabel(r'Phase angle $\alpha$')
ax.set_ylabel(r'Intensity ($L_{\odot}$)')
ax.legend()
fig.savefig('images/light_curves.jpg', bbox_inches="tight")

fig, ax = plt.subplots()
ax.plot(animation.alphas / np.pi, animation.planet_curve + animation.ring_curve + animation.star_curve,
        label='Ring + Planet + Star')
ax.xaxis.set_major_formatter(FuncFormatter(exoring_functions.format_fraction_with_pi))
ax.xaxis.set_major_locator(tck.MultipleLocator(base=1 / 2))
ax.set_xlabel(r'Phase angle $\alpha$')
ax.set_ylabel(r'Intensity ($L_{\odot}$)')
ax.legend()
fig.savefig('images/light_curves_with_star.jpg', bbox_inches="tight")

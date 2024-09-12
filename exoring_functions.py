import numpy as np


def circle_section_integral(radius, bounds: []):
    """The formula for the area of a circle section with sagitta lying on the x-axis"""
    upper = radius ** 2 * np.arcsin(bounds[1] / radius) + bounds[1] * np.sqrt(radius ** 2 - bounds[1] ** 2)
    bottom = radius ** 2 * np.arcsin(bounds[0] / radius) + bounds[0] * np.sqrt(radius ** 2 - bounds[0] ** 2)
    integration_result = upper - bottom
    return integration_result


def overlap_area(r_circle, r_ellipse, mu, cos_phi, sin_phi, offset):
    """
    Finds the overlap area of a circle with an ellipse

    Parameters
    ----------
        r_circle: (float) the radius of the circle
        r_ellipse: (float) the semi-major axis of the ellipse
        mu: (float) ratio of semi-minor axis over semi-major axis
        cos_phi: (float) cosine of angle phi between semi-major axis and line joining shape centers
        sin_phi: (float) sine of angle phi
        offset: (float) distance between centers of objects

    Returns
    -------
    Area of overlap area
    """

    sin_phi *= (-1 + 2 * (
            cos_phi >= 0))  # aligns everything with closest axis instead of same axis everytime - keeps bounds w. correct sign
    cos_phi = np.abs(cos_phi)

    def find_distance_from_ellipse_centre(a, b):
        with np.errstate(all='raise'):
            if mu == 0:
                # case of overlap of two circles
                return a ** 2 + b ** 2
            else:
                return (a * cos_phi + b * sin_phi) ** 2 + (1 / mu ** 2) * (a * sin_phi - b * cos_phi) ** 2

    # creating 1D area parameterizing edge of circle
    angles = np.linspace(0, 2 * np.pi, 5000)
    xs = r_circle * np.cos(angles) + offset
    ys = r_circle * np.sin(angles)

    # finding intersection points
    in_ellipse = (find_distance_from_ellipse_centre(xs, ys) < r_ellipse ** 2)
    intersect_bool = np.roll(in_ellipse, 1) != in_ellipse
    x = xs[intersect_bool]
    y = ys[intersect_bool]

    # coordinate transform - rotation such that semi-major axis aligns with x-axis plus stretch to circularize ellipse
    x_prime = (x * cos_phi + y * sin_phi)
    y_prime = (y * cos_phi - x * sin_phi) / mu

    ellipse_area = np.pi * mu * r_ellipse ** 2
    circle_area = np.pi * r_circle ** 2

    # checking different possibilities for different numbers of intersection points
    # no intersection
    if len(x) == 0:
        if np.all(in_ellipse):
            return circle_area
        elif not np.any(in_ellipse) and np.abs(offset) < r_circle:
            return ellipse_area
        else:
            return 0.

    # two intersection points
    elif len(x) == 2:

        # rotating circle and intersect points to use circle_section_integral in exoring_functions
        circle_rot_angle = np.arctan((x[1] - x[0]) / (y[1] - y[0]))
        circle_bound, extra = (x - offset) * np.cos(circle_rot_angle) - y * np.sin(circle_rot_angle)

        # catching numerical errors:
        if np.abs(circle_bound / r_circle) >= 1:
            if np.sign(circle_bound == np.sign(offset)):
                return min(ellipse_area, circle_area)
            else:
                return 0.

        circle_sign = np.sign(offset) * np.sign(offset - (x[0] - y[0] * (x[1] - x[0]) / (y[1] - y[0])))
        circle_section_area = np.abs(
            circle_section_integral(r_circle, bounds=[circle_sign * np.abs(circle_bound), r_circle]))

        ellipse_rot_angle = np.arctan2((x_prime[1] - x_prime[0]), (y_prime[1] - y_prime[0]))

        ellipse_bound, extra = x_prime * np.cos(ellipse_rot_angle) - y_prime * np.sin(ellipse_rot_angle)
        # catching numerical errors:
        if np.abs(ellipse_bound / r_ellipse) >= 1:
            if np.sign(ellipse_bound) != np.sign(offset):
                return min(ellipse_area, circle_area)
            else:
                return 0.

        ellipse_sign = np.sign(offset) * np.sign(x[0] - y[0] * (x[1] - x[0]) / (y[1] - y[0]))
        ellipse_section_area = mu * np.abs(
            circle_section_integral(r_ellipse, bounds=[ellipse_sign * np.abs(ellipse_bound), r_ellipse]))
        return ellipse_section_area + circle_section_area

    # four intersection points
    elif len(x) == 4:
        intersect_index = np.where(intersect_bool)[0]
        i_0 = intersect_index[0]
        # setting up which 2 points to take each time
        # we want two points where the section of ellipse is outside the section of circle
        clockwise_connected = in_ellipse[
            i_0 - 1]  # is the first point in the list paired with the one clockwise or anti-clockwise relative to it
        connection_direction = 1 - 2 * clockwise_connected  # the sign is relevant for finding the length of the circle section between two points
        if clockwise_connected:
            x = np.roll(x, 1)
            y = np.roll(y, 1)
            x_prime = np.roll(x_prime, 1)
            y_prime = np.roll(y_prime, 1)
            intersect_index = np.roll(intersect_index, 1)

        area_diff = 0
        for i in range(2):
            # selecting the correct pairs of intersection points
            x_i = x[2 * i:2 * (i + 1)]
            y_i = y[2 * i:2 * (i + 1)]
            x_i_prime = x_prime[2 * i:2 * (i + 1)]
            y_i_prime = y_prime[2 * i:2 * (i + 1)]
            index_i = intersect_index[2 * i:2 * (i + 1)]

            # check if the section of the circle subtending the two intersection points is less than the circumference
            # if yes the signs of the two bounds in the integral should be the same
            index_i = index_i[::connection_direction]
            if index_i[0] < index_i[1] and clockwise_connected:
                # python can't loop back around to the start when splicing arrays, or at least I don't know how
                circle_section_size = len(in_ellipse) - index_i[1] + index_i[0]
            else:
                circle_section_points = in_ellipse[index_i[0]:index_i[1]:connection_direction]
                circle_section_size = np.sum(circle_section_points)
            small_circle_section = circle_section_size < len(in_ellipse) / 2
            circle_bound_sign = 2 * small_circle_section - 1

            circle_rot_angle = np.arctan((x_i[1] - x_i[0]) / (y_i[1] - y_i[0]))
            circle_bound, extra = (x_i - offset) * np.cos(circle_rot_angle) - y_i * np.sin(circle_rot_angle)
            circle_section_area = np.abs(
                circle_section_integral(r_circle, bounds=[circle_bound_sign * np.abs(circle_bound), r_circle]))

            other_x_prime = x_prime[2 - 2 * i:int(4 / (i + 1))]
            if np.all(other_x_prime < x_i_prime):
                ellipse_radius_sign = 1
            elif np.all(other_x_prime > x_i_prime):
                ellipse_radius_sign = -1
            else:
                print('bad result')

            ellipse_rot_angle = np.arctan((x_i_prime[1] - x_i_prime[0]) / (y_i_prime[1] - y_i_prime[0]))
            ellipse_bound, extra = x_i_prime * np.cos(ellipse_rot_angle) - y_i_prime * np.sin(ellipse_rot_angle)
            ellipse_section_area = mu * np.abs(
                circle_section_integral(r_ellipse, bounds=[ellipse_bound, ellipse_radius_sign * r_ellipse]))

            area_diff -= ellipse_section_area
            area_diff += circle_section_area

        return ellipse_area + area_diff

    else:
        raise NotImplementedError('Edge case of %.i intersection points, how did you even do this??!' % len(x))


def select_best_result(results):
    lowest_NLL = np.inf
    best_fit = None
    for result in results:
        NLL = result[0]
        fit = result[1]
        if NLL < lowest_NLL:
            best_fit = fit
            lowest_NLL = NLL
    return lowest_NLL, best_fit

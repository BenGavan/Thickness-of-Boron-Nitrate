# -*- coding: utf-8 -*-
"""
Calculates the thickness of a boron nitrate sample by using a minimising chi squared fit of the expected transmission
coefficients for different electron energies.

// TODO: Add Logging
// TODO: Add multi-thread logging to save log to file.
// TODO: Add configuration file option.
// TODO: Add option/works for  more that 3 columns.

Ben Gavan - 22/11/19
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Constants
relative_permittivity = 4
permittivity_of_free_space = 0.00553
V_0 = 3.0
e = 1
d_1 = (1.2 * pow(e, 2) * np.log(2)) / (8 * np.pi * relative_permittivity * permittivity_of_free_space * V_0)
approx_single_layer_thickness = 3  # Angstrom
max_number_of_layers = 100


# MARK: - Entry point of the program/script
def main():
    """
    Start point of the script.
    Returns
    ------
    None
    """
    energies, transmission_coefficients, transmission_coefficient_errors = get_data()
    start_time = time.time_ns()

    sample_thickness, reduced_chi2 = calculate_minimized_chi2_fit(energies,
                                                                  transmission_coefficients,
                                                                  transmission_coefficient_errors)

    degrees_of_freedom = len(energies) - 1
    chi2 = reduced_chi2 * degrees_of_freedom

    sample_thickness_uncertainty = calculate_uncertainty_in_d_from_chi2(energies,
                                                                        transmission_coefficients,
                                                                        transmission_coefficient_errors,
                                                                        sample_thickness,
                                                                        chi2)

    number_of_layers = calculate_number_of_layers(sample_thickness)

    # Calculates time taken for calculations
    end_time = time.time_ns()
    time_taken = end_time - start_time

    print_separator()
    print('time taken {}s'.format(time_taken / pow(10, 9)))
    print('sample thickness, d = {} ± {}'.format(sample_thickness, sample_thickness_uncertainty))
    print('number of layers = {}'.format(number_of_layers))

    plot_energy_transmission_coefficients(energies, transmission_coefficients, transmission_coefficient_errors,
                                          sample_thickness, sample_thickness_uncertainty, number_of_layers)


# MARK: - Plotting
def plot_chi2_against_d(energies, transmission_coefficients, transmission_coefficient_errors):
    """
    PLots reduced chi squared against samplee thickness, d
    Parameters
    ----------
    energies : numpy.ndarray
    transmission_coefficients : numpy.ndarray
    transmission_coefficient_errors : numpy.ndarray
    Returns
    ------
    None
    """
    current_d = 0.5
    chi2s = np.array([])
    ds = np.array([])
    while current_d < 1.5:
        chi2 = calculate_reduced_chi_2(energies, transmission_coefficients, transmission_coefficient_errors, current_d)
        chi2s = np.append(chi2s, chi2)
        ds = np.append(ds, current_d)
        current_d += 0.001

    plt.figure()
    plt.title(r'Reduced $\chi^2$ against possible sample thickness, d')
    plt.xlabel('Possible sample thicknesses, d')
    plt.ylabel(r'$reduced \chi^2$')
    plt.plot(ds, chi2s, 'k.')
    plt.show()


def plot_energy_transmission_coefficients(energies, transmission_coefficients, transmission_coefficient_errors, d,
                                          d_uncertainty, number_of_layers):
    """
    Plots the transmission coefficients against particle energy data.
    Also plots the fit of least chi squared.
    Parameters
    ----------
    energies : numpy.ndarray
    transmission_coefficients : numpy.ndarray
    transmission_coefficient_errors : numpy.ndarray
    d : float
    d_uncertainty : float
    Returns
    ------
    None
    """
    xs = np.linspace(min(energies), max(energies), 10000)

    chi2 = calculate_reduced_chi_2(energies, transmission_coefficients, transmission_coefficient_errors, d)

    plt.figure()

    # Raw data points
    plt.errorbar(energies, transmission_coefficients, yerr=transmission_coefficient_errors, fmt='kx', capsize=2,
                 ecolor='#e6320e', elinewidth=1)

    plt.plot(xs, transmission_coefficient_function(d, xs), linestyle='-.')

    plt.title('Transmission coefficient against electron energy')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Transmission Coefficient')

    fit_equation = r'$T \approx \exp \left[ -\frac{2 (d_{2} - d_{1}) (2m)^{\frac{1}{2}}}{\hbar} \sqrt{\bar{V} - E} \right]$'
    plt.legend(['minimized $\chi^2$ fit of: {}'.format(fit_equation), 'raw data'], frameon=False, loc='upper left')

    plt.annotate('sample thickness = (${:.3f} \pm {:.3f})$\u212B,'.format(d, d_uncertainty), (0, 0), (0, -40),
                 xycoords='axes fraction', textcoords='offset points',
                 va='top')
    plt.annotate('reduced $\chi^2$ = {:.2f},'.format(chi2), (0, 0), (200, -40),
                 xycoords='axes fraction', textcoords='offset points',
                 va='top')
    plt.annotate(r'Number of Layers =  ~{:.3f} ($\approx {:.0f}$)'.format(number_of_layers, number_of_layers), (0, 0), (0, -60),
                 xycoords='axes fraction', textcoords='offset points',
                 va='top')

    plt.tight_layout()
    plt.savefig('transmission-against-energy-plot-BenGavan.png', ppi=300)
    plt.show()


# MARK: - Transmission coefficient function calculations
def transmission_coefficient_function(d, particle_energy):
    """
    Calculates the transmission coefficient for a given potential thickness (d) and particle energy.
    Parameters
    ----------
    d : float
    particle_energy : float or numpy.ndarray
    Returns
    ------
    transmission_coefficient : float
    """
    sqrt = np.sqrt(np.abs(average_potential(d) - particle_energy))
    return np.exp(-1.024634 * actual_potential_thickness(d) * sqrt)


def actual_potential_thickness(d):
    """
    Calculates the actual potential thickness (= d_2 - d_1) from the idealised potential thickness (d).
    Parameters
    ----------
    d : float
    Returns
    --------
    actual_potential_difference: float
    """
    numerator = 2.4 * pow(e, 2) * np.log(2)
    denominator = 8 * np.pi * relative_permittivity * permittivity_of_free_space * V_0
    return d - (numerator / denominator)


def average_potential(d):
    """
    Calculates the average potential for a given potential width.
    Parameters
    --------
    d : float
    Returns
    --------
    average_potential : float
    """
    numerator = 2 * 1.15 * lambda_d(d) * d
    denominator = d - (2 * d_1)
    return V_0 - ((numerator / denominator) * np.log(np.abs((d - d_1) / d_1)))


def lambda_d(d):
    """
    Calculates lambda for a given ideal potential thickness (d).
    Parameters
    --------
    d : float
    Returns
    --------
    lambda : float
    """
    return (pow(e, 2) * np.log(2)) / (8 * np.pi * relative_permittivity * permittivity_of_free_space * d)


def calculate_number_of_layers(total_thickness):
    """
    Calculates the number of layers of sample from the total thickness calculated from the transmission.
    Parameters
    --------
    total_thickness : float
    Returns
    --------
    number_of_layers : float
    uncertainty : float
    """
    number_of_layers = total_thickness / 3
    return number_of_layers


# MARK: - Plotting calculations.
def calculate_minimized_chi2_fit(energies, transmission_coefficients, transmission_coefficient_errors):
    """
    Calculates values for sample thickness (d) which minimises the reduced chi squared.
    Parameters
    ----------
    energies : numpy.ndarray
    transmission_coefficients : numpy.ndarray
    transmission_coefficient_errors : numpy.ndarray
    start_d : float
    Returns
    ------
    minimized_d : float
    best_reduced_chi2 : float
    """
    print_star_separator('Calculating minimized chi2 fit.')

    start_thickness, start_reduced_chi2 = calculate_start_thickness_candidate(energies, transmission_coefficients,
                                                                              transmission_coefficient_errors)

    current_thickness = start_thickness
    best_reduced_chi2 = start_reduced_chi2

    tolerance = pow(10, -8)
    difference = 1
    thickness_increment = 0.1
    print_separator()
    while difference > tolerance:
        reduced_chi2_plus = calculate_reduced_chi_2(energies, transmission_coefficients,
                                                    transmission_coefficient_errors, current_thickness + thickness_increment)

        reduced_chi2_minus = calculate_reduced_chi_2(energies, transmission_coefficients,
                                                     transmission_coefficient_errors, current_thickness - thickness_increment)

        print('current thickness = {}, thickness increment = {}, reduced chi2 (minus) = {}, reduced chi2 (plus) = {}'
              .format(current_thickness, thickness_increment, reduced_chi2_minus, reduced_chi2_plus))

        # If reduced chi2 is better when thickness is decreased, decreased thickness.
        if reduced_chi2_minus < best_reduced_chi2:
            difference = best_reduced_chi2 - reduced_chi2_minus
            best_reduced_chi2 = reduced_chi2_minus
            current_thickness -= thickness_increment
            continue
        # If reduced chi2 is better when thickness is increased, increase thickness.
        elif reduced_chi2_plus < best_reduced_chi2:
            difference = best_reduced_chi2 - reduced_chi2_plus
            best_reduced_chi2 = reduced_chi2_plus
            current_thickness += thickness_increment
            continue
        # If increment does not improve, decrease thickness_increment by order of magnitude.
        elif thickness_increment > pow(10, -10):
            thickness_increment = thickness_increment / 10
            continue
        break  # If sample_thickness can not be improved, finish loop.

    minimized_d = current_thickness - thickness_increment
    return minimized_d, best_reduced_chi2


def calculate_start_thickness_candidate(energies, transmission_coefficients, transmission_coefficient_errors):
    """
    Calculate sample thickness candidates to start hill climbing from that value
    Parameters
    ----------
    energies : numpy.ndarray
    transmission_coefficients : numpy.ndarray
    transmission_coefficient_errors : numpy.ndarray
    Returns
    --------
    thickness_candidate : float
    candidate_reduced_chi2 : float
    """
    print_star_separator('Calculating start thickness candidates')

    possible_start_chi2s = []  # thickness (float), reduced_chi2 (float)

    for layer_number in range(1, max_number_of_layers, 1):
        thickness_candidate = layer_number * approx_single_layer_thickness
        reduced_chi2_candidate = calculate_reduced_chi_2(energies, transmission_coefficients,
                                                         transmission_coefficient_errors, thickness_candidate)
        possible_start_chi2s.append([thickness_candidate, reduced_chi2_candidate])

    print_value('possible_start_chi2s', possible_start_chi2s)

    # Find the value for thickness that has the lowest reduced chi 2
    best_reduced_chi2_index = 0
    for i in range(len(possible_start_chi2s)):
        if possible_start_chi2s[best_reduced_chi2_index][1] > possible_start_chi2s[i][1]:
            best_reduced_chi2_index = i

    candidate_value = possible_start_chi2s[best_reduced_chi2_index][0]
    candidate_reduced_chi2 = possible_start_chi2s[best_reduced_chi2_index][1]

    return candidate_value, candidate_reduced_chi2


def calculate_uncertainty_in_d_from_chi2(energies, transmission_coefficients, transmission_coefficient_errors,
                                         optimized_d, minimized_chi2):
    """
    Calculates uncertainty in thickness of sample, d, by taking the value of d which is ~1 reduced chi squared greater
    than minimized chi squared.
    Parameters
    ----------
    energies : numpy.ndarray
    transmission_coefficients : numpy.ndarray
    transmission_coefficient_errors : numpy.ndarray
    optimized_d : float
    minimized_chi2 : float
    Returns
    --------
    uncertainty : float
    """
    target_chi2 = minimized_chi2 + 1

    lower_d = calculate_sample_thickness_for_target_chi2_test(energies, transmission_coefficients,
                                                              transmission_coefficient_errors, optimized_d,
                                                              target_chi2, is_lower=True)
    lower_d_sigma = optimized_d - lower_d
    starting_upper_d = optimized_d + lower_d_sigma
    upper_d = calculate_sample_thickness_for_target_chi2_test(energies, transmission_coefficients,
                                                              transmission_coefficient_errors, starting_upper_d,
                                                              target_chi2, is_lower=False)

    uncertainty = (upper_d - lower_d) / 2
    return uncertainty


def calculate_sample_thickness_for_upper_target_chi2(energies, transmission_coefficients,
                                                     transmission_coefficient_errors,
                                                     start_sample_thickness, target_chi2):
    """
    Calculates the upper value for sample thickness for a given target chi 2 (NOT reduced chi2).
    Parameters
    --------
    energies : numpy.ndarray
    transmission_coefficients : numpy.ndarray
    transmission_coefficient_errors : numpy.ndarray
    start_sample_thickness : float
    target_chi2 : float
    Returns
    --------
    sample_thickness : float
    """
    current_thickness = start_sample_thickness

    tolerance = pow(10, -9)
    difference = 99999
    d_increment = 0.1
    print_separator()

    was_below = None
    is_below = None

    while difference > tolerance:
        print_separator()

        current_chi2 = calculate_chi_2(energies, transmission_coefficients,
                                       transmission_coefficient_errors, current_thickness)

        difference = current_chi2 - target_chi2

        if was_below is None:
            was_below = difference < 0

        difference = np.abs(difference)
        print_star_separator(difference)
        if current_chi2 < target_chi2:
            current_thickness += d_increment
            is_below = True
        elif current_chi2 > target_chi2:
            current_thickness -= d_increment
            is_below = False
        else:
            break  # Target chi2 reached

        if is_below != was_below:
            d_increment = d_increment / 10

        was_below = is_below

    return current_thickness


def calculate_sample_thickness_for_target_chi2_test(energies, transmission_coefficients,
                                                    transmission_coefficient_errors,
                                                    start_sample_thickness, target_chi2, is_lower):
    """
    Calculates the upper value for sample thickness for a given target chi 2 (NOT reduced chi2).
    Parameters
    --------
    energies : numpy.ndarray
    transmission_coefficients : numpy.ndarray
    transmission_coefficient_errors : numpy.ndarray
    start_sample_thickness : float
    target_chi2 : float
    Returns
    --------
    sample_thickness : float
    """
    if is_lower:
        print_star_separator('Calculating lower value of sample thickness for target chi squared = {}, starting at d = {}.'
                             .format(target_chi2, start_sample_thickness))
    else:
        print_star_separator('Calculating upper value of sample thickness for target chi squared = {}, starting at d = {}.'
                             .format(target_chi2, start_sample_thickness))

    current_thickness = start_sample_thickness

    tolerance = pow(10, -10)
    difference = 99999
    d_increment = 0.1
    if is_lower:
        d_increment = -d_increment
    print_separator()

    was_below = None
    is_below = None

    while difference > tolerance:
        current_chi2 = calculate_chi_2(energies, transmission_coefficients,
                                       transmission_coefficient_errors, current_thickness)

        print('chi squared (d = {}) = {}'.format(current_thickness, current_chi2))

        difference = current_chi2 - target_chi2

        if was_below is None:
            was_below = difference < 0

        difference = np.abs(difference)
        if current_chi2 < target_chi2:
            current_thickness += d_increment
            is_below = True
        elif current_chi2 > target_chi2:
            current_thickness -= d_increment
            is_below = False
        else:
            break  # Target chi2 reached

        if is_below != was_below:
            d_increment = d_increment / 10

        was_below = is_below

    return current_thickness


def calculate_reduced_chi_2(energies, transmission_coefficients, transmission_coefficient_errors, d):
    """
    Calculates the reduced chi squared
    Parameters
    --------
    energies : numpy.ndarray
    transmission_coefficients : numpy.ndarray
    transmission_coefficient_errors : numpy.ndarray
    d : float
    Returns
    --------
    reduced_chi: float
    """
    chi2 = calculate_chi_2(energies, transmission_coefficients, transmission_coefficient_errors, d)
    degrees_of_freedom = len(energies) - 1
    reduced_chi2 = chi2 / degrees_of_freedom
    print('Reduced chi2 (d = {}) = {}'.format(d, reduced_chi2))
    return reduced_chi2


def calculate_chi_2(energies, transmission_coefficients, transmission_coefficient_errors, d):
    """
    Calculates the chi squared
    Parameters
    --------
    energies : numpy.ndarray
    transmission_coefficients : numpy.ndarray
    transmission_coefficient_errors : numpy.ndarray
    d : float
    Returns
    --------
    chi2: float
    """
    chi2 = 0
    for i in range(len(energies)):
        difference = transmission_coefficients[i] - transmission_coefficient_function(d, energies[i])
        error = transmission_coefficient_errors[i]
        chi2 += pow(difference / error, 2)

    return chi2


# MARK: - Reading and Validating data.
def get_data():
    """
    Opens, reads, and validates raw data file
    Returns
    --------
    transmission_coefficients : np.array([float])
    energies : np.array([float])
    transmission_coefficient_errors : np.array([float])
    """

    raw_data_file = open_data_file()

    transmission_coefficients = np.array([])
    energies = np.array([])
    transmission_coefficient_errors = np.array([])

    for line in raw_data_file:
        #  If current file line is commented out using '%', ignore and continue reading file.
        if is_file_line_comment(line):
            continue

        # Splits line into line elements.
        line_elements = line.split(',')

        # Checks current line has correct # elements.
        if len(line_elements) != 3:
            print('ERROR: Line does not have 3 comma separated elements')
            return None, None, None

        # Checks input data is of type 'float' and checks if value is physical - if it's not, skip that data point.
        transmission_coefficient, is_float = cast_to_float(line_elements[0])
        if not is_float:
            continue
        if not is_transmission_coefficient_physical(transmission_coefficient):
            continue

        energy, is_float = cast_to_float(line_elements[1])
        if not is_float:
            continue
        if not is_energy_physical(energy):
            continue

        transmission_coefficient_error, is_float = cast_to_float(line_elements[2])
        if not is_float:
            continue

        # Appends current casted (and physically validated) data to complete data arrays.
        transmission_coefficients = np.append(transmission_coefficients, transmission_coefficient)
        energies = np.append(energies, energy)
        transmission_coefficient_errors = np.append(transmission_coefficient_errors, transmission_coefficient_error)

    energies, transmission_coefficients, transmission_coefficient_errors = remove_anomalous_values(energies,
                                                                                                   transmission_coefficients,
                                                                                                   transmission_coefficient_errors)

    return energies, transmission_coefficients, transmission_coefficient_errors


def open_data_file():
    """
    Opens data file, and if cannot find file, ask user to specify where.
    Returns
    --------
    raw_data_file : IO
    """
    filepath = 'data.csv'
    while True:
        try:
            raw_data_file = open(filepath, 'r')
            return raw_data_file
        except FileNotFoundError:
            filepath = input('File not found, please enter file path (relative to this script):')
            continue


def is_energy_physical(energy):
    """
    Check if energy is physical
    Parameters
    --------
    energy : float
    Returns
    --------
    is_energy_physical : bool
    """
    if energy >= 0:
        return True
    else:
        return False


def is_transmission_coefficient_physical(trans_coeff):
    """
    Checks if transmission coefficient (fraction of particles passed through sample) is physical - between 0 and 1
    Parameters
    --------
    trans_coeff : float
    Returns
    --------
    is_transmission_coefficient_physical : bool
    """
    if 0 <= trans_coeff <= 1:
        return True
    else:
        return False


def remove_anomalous_values(xs, ys, y_errs):
    """
    Removes the anomalous values from xs and ys based on whether they are withing 3 standard deviation.
    Parameters
    --------
    xs : ndarray
    ys : ndarray
    y_errs : ndarray
    Returns
    --------
    new_xs : numpy.ndarray
    new_ys : numpy.ndarray
    new_y_errs : numpy.ndarray
    """
    # Check given data columns are the same length.
    if (len(xs) != len(ys)) or (len(xs) != len(y_errs)):
        print('ERROR: length of arrays do not match')
        return

    std_ys = np.std(ys)
    std_xs = np.std(xs)

    mean_xs = float(np.mean(xs))  # TODO: decide of float or np.float64
    mean_ys = float(np.mean(ys))

    new_xs = np.array([])
    new_ys = np.array([])
    new_y_errs = np.array([])

    # Checks values are withing 3 standard deviation of the mean.
    for x, y, y_err in zip(xs, ys, y_errs):
        if is_in_range_of_std(x, mean_xs, std_xs) and is_in_range_of_std(y, mean_ys, std_ys) \
                and is_in_range_of_std(y_err, np.mean(y_errs), np.std(y_errs)):
            new_xs = np.append(new_xs, x)
            new_ys = np.append(new_ys, y)
            new_y_errs = np.append(new_y_errs, y_err)

    return new_xs, new_ys, new_y_errs


def is_in_range_of_std(value, mean, std):
    """
    Checks if the current value is within 3 standard deviations from the mean
    Parameters
    --------
    value : float
    mean : float
    std : float
    Returns
    --------
    is_in_range_of_std : bool
    """
    if (mean + std * 3) >= value >= (mean - std * 3):
        return True
    return False


def is_file_line_comment(line):
    """
    Checks if the line is commented out by checking if first character equal to '%'
    Parameters
    ----------
    line : str
    Returns
    ----------
    is_comment : bool
    """
    if len(line) != 0 and line[0] == '%':
        return True
    return False


#
# def is_float(value):
#     """
#     Checks if given value is a float.  Returns 'True' if object passed can be cast to float, returns 'False' is cast
#     fails via 'ValueError'.
#     Parameters
#     ----------
#     value : Any
#     Returns
#     ----------
#     is_float : bool
#     """
#     try:
#         float(value)
#         return True
#     except ValueError:
#         return False


def cast_to_float(value):
    """
    Attempts to cast value to float and returns the value and 'True' if successful and
    the value and 'False' if unsuccessful.
    Parameters
    ----------
    value : any
    Returns
    ----------
    value : float or any
    is_float : bool
    Created by
    ----------
    Ben Gavan 17/10/19
    """
    try:
        return float(value), True
    except ValueError:
        return value, False


def calculate_mean(values):
    values_sum = np.sum(values)
    return values_sum / len(sum)


def multiply_uncertainty(value_one, uncertainty_one, value_two, uncertainty_two, new_value):
    """
    Calculates the uncertainty (of new_value) when two values, both with uncertainty, are multiplied together.
    Parameters
    ----------
    value_one : float
    uncertainty_one : float
    value_two : float
    uncertainty_two : float
    new_value : float
    Returns
    ----------
    uncertainty : float
    Created by
    ----------
    Ben Gavan 17/10/19
    """
    return pow(pow(uncertainty_one / value_one, 2) + pow(uncertainty_two / value_two, 2), 0.5) * abs(new_value)


# MARK: - Debug utils.
def print_separator():
    """
    Prints line separator
    """
    print('----------------------------------')


def print_star_separator(label=''):
    """
    Prints star separator along with a value and label
    Parameters
    ----------
    label : str
    """
    if label != '':
        label = ' {} '.format(label)
    print("************{}************".format(label))


def print_value(label, value):
    """
    Prints separator along with a value and label
    Parameters
    ----------
    label : str
    value : str
    """
    print_separator()
    print(label)
    print(value)


if __name__ == '__main__':
    main()

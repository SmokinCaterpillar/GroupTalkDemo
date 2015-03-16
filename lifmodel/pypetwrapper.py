__author__ = 'Robert Meyer'

import os
import numpy as np
import pandas as pd
import logging

from pypet import Parameter, cartesian_product, Environment

from lifneuron import euler_neuron


def pypet_neuron(traj):
    """ Wraps the `euler_neuron` function

    :param traj: Trajectory container with all parameters

    :return: Estimate of firing rate.

    """
    parameter_dict = traj.parameters.f_to_dict(short_names=True, fast_access=True)
    V_array, spiketimes, times = euler_neuron(**parameter_dict)

    # Calculate the spikes
    nspikes = len(spiketimes)

    traj.f_add_result('neuron.$', V=V_array, nspikes=nspikes, spiketimes=spiketimes,
                      comment='Contains the development of the membrane potential over time '
                              'as well as the number of spikes and the spikestimes.')

    # * 1000 because we assume units of milliseconds
    return nspikes/traj.duration * 1000.0


def neuron_postproc(traj, result_list):
    """Postprocessing, sorts computed firing rates into a table

    :param traj:

        Container for results and parameters

    :param result_list:

        List of tuples, where first entry is the run index and second is the actual
        result of the corresponding run.

    :return:
    """

    # Let's create a pandas DataFrame to sort the computed firing rate according to the
    # parameters. We could have also used a 2D numpy array.
    # But a pandas DataFrame has the advantage that we can index into directly with
    # the parameter values without translating these into integer indices.
    I_range = traj.par.neuron.f_get('I').f_get_range()
    ref_range = traj.par.neuron.f_get('tau_ref').f_get_range()

    I_index = sorted(set(I_range))
    ref_index = sorted(set(ref_range))
    rates_frame = pd.DataFrame(columns=ref_index, index=I_index)
    # This frame is basically a two dimensional table that we can index with our
    # parameters

    # Now iterate over the results. The result list is a list of tuples, with the
    # run index at first position and our result at the second
    for result_tuple in result_list:
        run_idx = result_tuple[0]
        firing_rates = result_tuple[1]
        I_val = I_range[run_idx]
        ref_val = ref_range[run_idx]
        rates_frame.loc[I_val, ref_val] = firing_rates # Put the firing rate into the
        # data frame

    # Finally we going to store our new firing rate table into the trajectory
    traj.f_add_result('summary.firing_rates', rates_frame=rates_frame,
                      comment='Contains a pandas data frame with all firing rates.')


def add_parameters(traj):
    """Adds all parameters to `traj`"""
    print('Adding Results')

    traj.f_add_parameter('neuron.V_init', 0.0,
                         comment='The initial condition for the '
                                    'membrane potential')
    traj.par = Parameter('neuron.I', 0.0,
                         comment='The externally applied current.')

    traj.par.neuron.tau_V =  12.0, 'The membrane time constant in milliseconds'

    traj.par.neuron.tau_ref = 5.0, 'The refractory period in milliseconds '

    traj.f_add_parameter_group('simulation')

    traj.simulation.duration = 1000.0, 'The duration of the experiment in milliseconds.'
    traj.simulation.dt =  0.1, 'The step size of an Euler integration step.'


def add_exploration(traj):
    """Explores different values of `I` and `tau_ref`."""

    print('Adding exploration of I and tau_ref')

    explore_dict = {'neuron.I': np.arange(0, 1.01, 0.05).tolist(),
                    'neuron.tau_ref': [5.0, 7.5, 10.0]}

    explore_dict = cartesian_product(explore_dict, ('neuron.tau_ref', 'neuron.I'))
    # The second argument, the tuple, specifies the order of the cartesian product,
    # The variable on the right most side changes fastest and defines the
    # 'inner for-loop' of the cartesian product

    traj.f_explore(explore_dict)


def main():

    filename = os.path.join('hdf5', 'FiringRate.hdf5')
    env = Environment(trajectory='FiringRate',
                      comment='Experiment to measure the firing rate '
                            'of a leaky integrate and fire neuron. '
                            'Exploring different input currents, '
                            'as well as refractory periods',
                      add_time=False, # We don't want to add the current time to the name,
                      log_stdout=True,
                      multiproc=True,
                      ncores=2, #My laptop has 2 cores ;-)
                      wrap_mode='QUEUE',
                      filename=filename,
                      overwrite_file=True)

    traj = env.v_trajectory

    # Add parameters
    add_parameters(traj)

    # Let's explore
    add_exploration(traj)

    # Ad the postprocessing function
    env.f_add_postprocessing(neuron_postproc)

    # Run the experiment
    env.f_run(pypet_neuron)

    # Finally disable logging and close all log-files
    env.f_disable_logging()


if __name__ == '__main__':
    main()
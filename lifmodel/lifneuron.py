__author__ = 'Robert Meyer'

import numpy as np
import matplotlib.pyplot as plt

def euler_neuron(V_init, I, tau_V, tau_ref, dt, duration):
    """ Simulation of a leaky integrate and fire neuron.

    Simulates the equation

        tau_V * dV/dT = -V + I

    with a simple Euler scheme.

    This is a unitless system, reset is performed for V >= 1.
    Voltage is clamped for a refractory period after reset.

    :param V_init: Initial value
    :param I: Input current
    :param tau_V: Membrane time constant
    :param tau_ref: Refractory period
    :param dt: Timestep
    :param duration: runtime

    :return:

        The development V over time (numpy array)
        A list of all spiketimes
        and the corresponding simulation times (numpy array)
        .
    """
    # Createa a times array
    times = np.arange(0, duration, dt)
    steps = len(times)

    # Create the V array for the membrane development over time
    V_array = np.zeros(steps)
    V_array[0] = V_init
    spiketimes = []  # List to collect all times of action potentials

    # Do the Euler integration:
    print 'Starting Euler Integration'
    for step in range(1, steps):
        if V_array[step-1] >= 1:
            # The membrane potential crossed the threshold and we mark this as
            # an action potential
            V_array[step] = 0
            spiketimes.append(times[step-1])
        elif spiketimes and times[step] - spiketimes[-1] <= tau_ref:
            # We are in the refractory period, so we simply clamp the voltage
            # to 0
            V_array[step] = 0
        else:
            # Euler Integration step:
            dV = -1.0 / tau_V * V_array[step-1] + I
            V_array[step] = V_array[step-1] + dV * dt
    print 'Finished Euler Integration'

    return V_array, spiketimes, times


def main():
    V_init = 0.0
    I = 0.1
    tau_V = 12.0
    tau_ref = 5.0
    dt = 0.1
    duration = 250.0

    V_array, spiketimes, times = euler_neuron(V_init, I, tau_V, tau_ref, dt, duration)

    plt.plot(times, V_array)
    plt.show()


if __name__ == '__main__':
    main()
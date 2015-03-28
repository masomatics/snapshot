"""
data_generation.py

"""

import numpy as np


def generate_ar(init_mean, theta, T, sig_sys, sig_obs, N,
                sys_seed=9999, obs_seed=9998):
    """
    Generate a size N batch of simulation paths

    x(t+1) = f(x(t-1)) + N(0, diag(\sigma_{sys}^2))
    y(t) = x(t)  + N(0, diag(\sigma_{noise})^2)

    x_{a}(t+1) =  x_{a}(t) + \theta_{0} \frac{x_{a}(t)^2}{1 + x_{a}(t)^2} - theta_{1} x_{c}^2 + N(0, \sigma_{a}^2))
    x_{b}(t+1) =  x_{b}(t) + \theta_{2} x_{a}(t) - \theta_{3} x_{b}(t) + N(0, \sigma_{b}^2)
    x_{c}(t+1) =  x_{c}(t) + \theta_{4} x_{b}(t) - \theta_{5} x_{c}(t) + N(0, \sigma_{c}^2)

    Parameters
    ----------
    init_mean: 1-d ndarray (num_parameters, ) float64
    theta: 1-d ndarray (num_species, ) float64
    T: int
    sys_seed: int random seed for system noise
    obs_seed: int random seed for observation noise
    sig_sys: 1-d ndarray sigma of system noise (diag)
    sig_obs: 1-d ndarray sigma of observation noise (diag)

    Variables
    ---------
    x_now: 2-d ndarary (N, num_species) float64
    y_timeseries: 3-d ndarray (T, N, num_species) float64

    Return
    ------
    y_timeseries

    Example
    -------

    """
    # fixed parameters
    num_parameters = len(theta)
    num_species = len(init_mean)

    # noise generation
    np.random.seed(sys_seed)
    sys_noise = sig_sys * np.random.randn(T, N, num_species)  # gaussian noise
    np.random.seed(obs_seed)
    obs_noise = sig_obs * np.random.randn(T, N, num_species)  # gaussian noise

    # initialization
    x_now = np.array([init_mean for _ in range(N)], np.float64)
    y_timeseries = np.zeros((T, N, num_species), np.float64)
    y_timeseries[0] = x_now + obs_noise[0, :, :]

    for t in range(1, T):
        # generate x
        x_now[:, 0] = x_now[:, 0] \
                      + theta[0] * (x_now[:, 1] ** 2) / (1 + (x_now[:, 0] ** 2)) \
                      - theta[1] * (x_now[:, 2] ** 2)

        x_now[:, 1] = x_now[:, 1] \
                      + theta[2] * x_now[:, 0] \
                      - theta[3] * x_now[:, 1]

        x_now[:, 2] = x_now[:, 2] \
                      + theta[4] * x_now[:, 1] \
                      - theta[5] * x_now[:, 2]

        x_now += sys_noise[t, :, :]
        x_now = np.maximum(x_now, 0)

        # generate y
        y_timeseries[t, :, :] = np.maximum(x_now + sys_noise, 0)

    return y_timeseries

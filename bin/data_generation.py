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
    N: notice! num_x = num_y = N (?)

    Variables
    ---------
    x_now: 2-d ndarary (N, num_species) float64
    y_timeseries: 3-d ndarray (T, N, num_species) float64

    Return
    ------
    y_timeseries

    Example
    -------
    >>> import matplotlib.pyplot as pltx
    >>> init_mean = np.array([2., 1., 2.])
    >>> theta = np.array([0.5 , 2., 1., 0.75, 0.75, 1.]) * 0.1
    >>> T = 20
    >>> sig_sys = np.array([2., 1., 1.])
    >>> sig_obs = np.array([1., 1., 1.])
    >>> N = 10000
    >>> plt.plot()
    >>> plt.plot(y.mean(axis=1)[:, 0], label='1')
    >>> plt.plot(y.mean(axis=1)[:, 1], label='2')
    >>> plt.plot(y.mean(axis=1)[:, 2], label='3')
    >>> plt.legend()
    >>> plt.show()

    """
    # check parameter size
    num_parameters = 6
    num_species = 3
    assert(init_mean.shape[0] == num_species)
    assert(theta.shape[0] == num_parameters)

    # noise generation
    np.random.seed(sys_seed)
    sys_noise = sig_sys * np.random.randn(T, N, num_species)
    np.random.seed(obs_seed)
    obs_noise = sig_obs * np.random.randn(T, N, num_species)

    # initialization
    x_now = np.array([init_mean for _ in range(N)], np.float64)
    y_timeseries = np.zeros((T, N, num_species), np.float64)
    y_timeseries[0] = x_now + obs_noise[0, :, :]

    for t in range(1, T):
        # generate x
        x_now[:, 0] = x_now[:, 0] \
                      + theta[0] * (x_now[:, 0] ** 2) / (1 + (x_now[:, 0] ** 2)) \
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
        y_timeseries[t, :, :] = np.maximum(x_now + obs_noise[t, :, :], 0)


    return y_timeseries

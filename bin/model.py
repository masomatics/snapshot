"""
deriv_ar.py

"""

import numpy as np


class Model(object):

    def __init__(self, init_mean_x, T, y_obs, y_obs_times, num_x):
        """
        Params
        ------
        init_mean_x: 1-d ndarray (num_params, ) float64 initial mean of x
        y_obs: (num_y_obs, num_y, dim) (num_y_obs is # of snapshots, dim_x == dim_y, num_y is # of samples at t)
        y_obs_times (num_y_obs, )

        """

        # set attributes
        self.init_mean_x = init_mean_x
        self.T = T
        self.y_obs = y_obs
        self.y_obs_times = y_obs_times
        self.num_x = num_x

        # dim
        assert(y_obs.shape[0] == y_obs_times.shape[0])
        self.num_y_obs, self.num_y, self.dim = y_obs.shape
        self.num_params = theta.shape[0]

    def get_deriv_and_energy(self, theta, sys_seed, sys_obs):
        pass

    def _update_x(self, theta):
        pass


class NonlinearAR(Model):
    """
    x(t+1) = f(x(t-1)) + N(0, diag(\sigma_{sys}^2))
    y(t) = x(t)  + N(0, diag(\sigma_{noise})^2)

    x_{a}(t+1) =  x_{a}(t) + \theta_{0} \frac{x_{a}(t)^2}{1 + x_{a}(t)^2} - theta_{1} x_{c}^2 + N(0, \sigma_{a}^2))
    x_{b}(t+1) =  x_{b}(t) + \theta_{2} x_{a}(t) - \theta_{3} x_{b}(t) + N(0, \sigma_{b}^2)
    x_{c}(t+1) =  x_{c}(t) + \theta_{4} x_{b}(t) - \theta_{5} x_{c}(t) + N(0, \sigma_{c}^2)

    x(t+1) = x(t) + A

    """

    def __init__(self, init_mean_x, T, y_obs, y_obs_times, num_x, sig_sys, sig_obs):
        """
        sig_sys: 1-d ndarray sigma of system noise (diag) (dim)
        sig_obs: 1-d ndarray sigma of observation noise (diag) (dim)

        """
        super(NonlinearAR, self).__init__(init_mean_x, T, y_obs, y_obs_times, num_x)

        self.sig_sys = sig_sys  # (dim, )
        self.sig_obs = sig_obs  # (dim, )

        self.theta_coeff = np.zeros((self.num_params, self.dim, self.num_x))

    def get_deriv_and_energy(self, theta, sys_seed, obs_seed):
        """
        deriv_r = (num_params, )
        E_p_mk = (num_y, )
        p_y_mkj = (num_x, )
        deriv_E_p_kr = (num_x, num_y)
        deriv_log_p_x = (num_params, num_x)

        """
        super(NonlinearAR, self).get_deriv_and_energy(theta, sys_seed, obs_seed)

        self.theta = theta

        # initialization
        np.random.seed(sys_seed)
        self.sys_noise = self.sig_sys * np.random.randn(self.T, self.num_x, self.dim)
        np.random.seed(obs_seed)
        self.obs_noise = self.obs_sys * np.random.randn(self.T, self.num_y, self.dim)

        self.x_now = self.init_x + sys_noise[0]  # (num_x, dim)
        deriv_log_p_x = np.zeros(num_params, num_x)
        m = 0  # index of y_obs_times

        E_p_list = []
        deriv_list = []

        # monte carlo simulation
        for t in range(1, self.T):
            # compute derivative of logP(x[0, t_m))
            deriv_log_p_x += self._compute_deriv_log_p_x_at_t(t)  # (num_x, num_params)

            if (m is not None) and (t == self.y_obs_times[m]):
                # p_y_mkj (num_x, num_y), max_d (num_y, )
                p_y_mkj, max_d = self._compute_p_y_mkj(t, m)
                # E_p_mk (num_y, )
                E_p_mk = self._compute_E_p_mk(p_y_mkj)
                # deriv_E_p_kr (num_y, num_params)
                deriv_E_p_kr = self._compute_deriv_E_p_kr(p_y_mkj, deriv_log_p_x)
                # store E_p_mk for calculate energy
                E_p_list.append(E_p_mk)
                # store deriv each time
                deriv_at_t = (deriv_E_p_kr / E_p_mk).sum(axis=0)
                deriv_list.append(deriv_at_t)
                # update m (index of obs_times)
                m += 1 if m + 1 < num_t_obs else None

            # update x_now
            x_now = self.update_x(x_now, theta, sys_noise[t])
            ## x_now = self._update_x()

        # calculate derivative
        deriv = self._compute_deriv(deriv_list)
        # calculate energy
        energy = self._compute_energy(E_p_list)

        return deriv, energy

    def _update_x(self, theta):
        super(NonlinearAR, self)._update_x(theta)

    def _compute_deriv_log_p_x_at_t(self, t):
        # set
        A = np.array([[1, -1,  0,  0,  0,  0],
                      [0,  0,  1, -1,  0,  0],
                      [0,  0,  0,  0,  1, -1]], np.float64)
        x = self.x_now  # (num_x, )
        e = self.sys_noise[t]  # (num_x, dim)
        s = self.sig_sys  # (dim, )
        deriv_theta = np.zeros((self.num_x, self.num_params),
                               np.float64)  # (num_x, num_params)

        # (num_x, ) * np.dot((dim, num_x), (dim, )) -> (num_x, )
        deriv_theta[:, 0] = (x[:, 0] ** 2 / (1 + x[:, 0] ** 2)) \
                            * np.dot(e / s, A[:, 0])
        deriv_theta[:, 1] = (x[:, 1] ** 2) * np.dot(e / s, A[:, 1])
        deriv_theta[:, 2] = x[:, 2] * np.dot(e / s, A[:, 2])
        deriv_theta[:, 3] = x[:, 3] * np.dot(e / s, A[:, 3])
        deriv_theta[:, 4] = x[:, 4] * np.dot(e / s, A[:, 4])
        deriv_theta[:, 5] = x[:, 5] * np.dot(e / s, A[:, 5])

        return deriv_theta  # (num_x, num_params)

    def _compute_p_y_mkj(self, t, m):
        y = self.y_obs[m]  # (num_y, dim)
        x = self.x_now  # (num_x, dim)
        s = self.sig_obs  # (dim, )
        dist = x.reshape((num_x, 1, dim)) - y.reshape(1, num_x, dim)  # (num_x, num_y, dim)
        max_d = dist.max(axis=0)  # (num_y, dim)
        # 1 / sqrt(2 \pi \sigma) is omitted because of cancel
        H = 0.5 * np.dot((dist - max_d.reshape(1, num_y, dim)) ** 2, np.power(s, -1))  # (num_x, num_y)
        p_y_mkj = np.exp(- H)  # (num_x, num_y)

        return p_y_mkj, max_d  # (num_x, num_y), (num_y, )

    def _compute_E_p_mk(self, p_y_mkj):
        E_p_mk = p_y_mkj.mean(axis=0)  # (num_y, )
        return E_p_mk  # (num_y, )

    def _compute_deriv_E_p_kr(self, p_y_mkj, deriv_log_p_x):
        # np.dot((num_x, num_y).T, (num_x, num_params)) -> (num_y, num_params)
        deriv_E_p_kr = 1. / self.num_x * np.dot(p_y_mkj.T, deriv_log_p_x)
        return deriv_E_p_kr  # (num_y, num_params)

    def _compute_deriv(self, deriv_list):
        """
        \sum_{t_m} \sum_{y_k} \frac{\partial}{\partial \theta} \log E_{x} \left\[ p(y_k | x)  \right\]
        """
        ret = np.zeros(num_parameters)
        for d in deriv_list:
            ret += d
        return ret


    def _compute_engergy(self, E_p_list):
        """
        Negative Log Likelihood

        - \sum_{t_m} \sum_{y_k} \log E_{x} \left\[ p(y_k | x)  \right\]

        """
        return sum([np.log(E_p).sum() for E_p in E_p_list]) / self.num_y_obs


"""
num_t_obs = num_slices,
dim_state = num_species,
num_y = num_particles
num_x = N
E_p_mk = tilde_p_ymk
deriv_E_p_kr = dEP_kr

"""

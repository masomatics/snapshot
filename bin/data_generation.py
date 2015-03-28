"""
data_generation.py

"""


def generate_ar():
    """
    Generate a size N batch of simulation paths
    This for the discrete process used in the paper of Doucet.

    x(m) = f(x(m-1)) + N(0, sigW^2)
    y(m) = x(m)  + N(0, sigW^2)

    xA(m) =  theta1  xC(m-1)^2 - theta2 xA(m-1)^2  + N(0, sigW_A^2)
    xB(m) =  theta3  xA(m-1) - theta4 xB(m) +  N(0, sigW_B^2)
    xC(m) =  theta5  xB(m-1) - theta6 xC(m) +  N(0, sigW_C^2)

    Input
    double[dimen] initmean
    int  T >0
    Covariance Matrix sigV
    Covariance Matrix sigW
    double[num_species][N][num_timepts] rnsource_sys
    double[num_species][N][num_timepts] rnsource_obs
    int N number of experiments

    """
    pass

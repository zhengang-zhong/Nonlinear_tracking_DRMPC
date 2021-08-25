import numpy as np
from DRO_class import Opt_problem, Simulation
import casadi as ca
import mosek
import cvxpy as cp


def mass_spring_ode(x, u):
    m = 2  # [kg]
    k1 = 3  # [N/m]
    k2 = 2  # [N/m]

    A = ca.DM([[0, 1], [-k2 / m, -k1 / m]])
    B = ca.DM([[0], [1 / m]])

    dot_x = A @ x + B @ u

    return dot_x

if __name__ == "__main__":
    x_SX = ca.SX.sym("x_SX", 2)
    u_SX = ca.SX.sym("u_SX", 1)

    ode = ca.Function("ode_func", [x_SX, u_SX], [mass_spring_ode(x_SX, u_SX)])

    C = np.array([[1e-3, 0], [0, 0]])
    # C = np.array([[0,0], [0, 0]])
    D = np.array([[1, 0]])
    E = np.array([[0, 1e-3]])
    # E = np.array([[0,0]])
    N = 3

    delta_t = 0.1

    x_init = np.array([[-2], [0]])
    # x_init = np.array([[0], [0]])
    # x_init = np.array([[ 2.50000775e-01],[-4.99584460e-07]])
    # u_0 = np.array([[0.5]])

    # x_init = np.array([[ 3.00000930e-01],[-5.99501352e-07]])
    # u_0 = np.array([[0.6]])
    u_0 = np.array([[0]])

    # xr = np.array([[0], [0]])
    # ur = np.array([[0.5]])
    # ur = np.array([[0]])

    # xr = np.array([[ 2.50000775e-01],[-4.99584460e-07]])
    # ur = np.array([[0.5]])
    xr = np.array([[0],[0]])
    ur = np.array([[0]])
    # xr = np.array([[ 3.00000930e-01],[-5.99501352e-07]])
    # ur = np.array([[0.6]])

    Q = np.diag([10, 1])
    R = np.array([[1]])
    Qf = np.diag([15, 1])

    mu = np.zeros([2, 1])
    sigma = 1
    sin_const = 1
    mass_sim = Simulation(ode, delta_t, N, x_init, C, D, E, Q, Qf, R, cont_time=True, nonlinear=False, u_0=u_0,
                          xr=xr, ur=ur, collect=False, est=False, mu=mu, sigma=sigma, beta=1,
                          sin_const=sin_const, N_sample=1, epsilon=1, i_th_state=1, i_state_ub=0.5, N_sim=200,
                          mode="gene", data_set=None, N_sample_max=None)

    mass_sim.plot_state()
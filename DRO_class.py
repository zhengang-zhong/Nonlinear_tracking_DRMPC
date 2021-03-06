import numpy as np
import casadi as ca
import cvxpy as cp
import mosek
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
import gurobipy

class Model:
    def __init__(self, sys_fn, delta_t, cont_time=True, nonlinear=False):
        self.sys_fn = sys_fn
        self.delta_t = delta_t
        self.cont_time = cont_time
        self.nonlinear = nonlinear

        self.Nx = sys_fn.sx_in()[0].shape[0]
        self.Nu = sys_fn.sx_in()[1].shape[0]

        self.xk_SX = ca.SX.sym("xk_SX", self.Nx)
        self.uk_SX = ca.SX.sym("uk_SX", self.Nu)

        if cont_time == True:
            self.dt_sys_fn = self.discretize_sys()
        else:
            self.dt_sys_fn = self.sys_fn

    def discretize_sys(self):
        xk_SX = self.xk_SX
        uk_SX = self.uk_SX
        sys_fn = self.sys_fn
        delta_t = self.delta_t
        x_next = self.integrator_rk4(sys_fn, xk_SX, uk_SX, delta_t)
        dt_sys_fn = ca.Function("dt_sys_fn", [xk_SX, uk_SX], [x_next])
        return dt_sys_fn

    def integrator_rk4(self, f, x, u, delta_t):
        '''
        This function calculates the integration of stage cost with RK4.
        '''

        k1 = f(x, u)
        k2 = f(x + delta_t / 2 * k1, u)
        k3 = f(x + delta_t / 2 * k2, u)
        k4 = f(x + delta_t * k3, u)

        x_next = x + delta_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next


class Linear_model(Model):
    '''
    Linearized model
    '''

    def __init__(self, sys_fn, delta_t, cont_time=True, nonlinear=False):
        super().__init__(sys_fn, delta_t, cont_time, nonlinear)

        x0_SX = ca.SX.sym("x0_SX", self.Nx)
        u0_SX = ca.SX.sym("u0_SX", self.Nu)
        self.x0_SX = x0_SX
        self.u0_SX = u0_SX

        xk_SX = self.xk_SX
        uk_SX = self.uk_SX
        self.lin_dt_sys_fn = self.linearize_sys(x0_SX, xk_SX, u0_SX, uk_SX)

    def linearize_sys(self, x0_SX, xk_SX, u0_SX, uk_SX):
        dt_sys = self.dt_sys_fn(xk_SX, uk_SX)
        A = ca.jacobian(dt_sys, xk_SX)
        B = ca.jacobian(dt_sys, uk_SX)

        A_fn = ca.Function("A_fn", [xk_SX, uk_SX], [A])
        B_fn = ca.Function("B_fn", [xk_SX, uk_SX], [B])

        self.A_fn = A_fn
        self.B_fn = B_fn

        if self.nonlinear == True:
            x_next = self.dt_sys_fn(x0_SX, u0_SX) + A_fn(x0_SX, u0_SX) @ (xk_SX - x0_SX) + B_fn(x0_SX, u0_SX) @ (
                        uk_SX - u0_SX)
            x_next_lin_fn = ca.Function("x_next_lin_fn", [x0_SX, xk_SX, u0_SX, uk_SX], [x_next])
            self.delta_fn = ca.Function("delta_fn", [x0_SX, u0_SX], [
                self.dt_sys_fn(x0_SX, u0_SX) - A_fn(x0_SX, u0_SX) @ x0_SX - B_fn(x0_SX, u0_SX) @ u0_SX])
            self.A_fn = A_fn
            self.B_fn = B_fn
        else:
            # Linear system is independent of x0 and u0
            x_next = A_fn(xk_SX, uk_SX) @ xk_SX + B_fn(xk_SX, uk_SX) @ uk_SX
            x_next_lin_fn = ca.Function("x_next_lin_fn", [xk_SX, uk_SX], [x_next])
            self.delta_fn = ca.Function("delta_fn", [x0_SX, u0_SX], [np.zeros(xk_SX.shape)])
            self.delta = np.zeros(xk_SX.shape)
            self.A = ca.DM(A)  # Transfer SX to DM
            self.B = ca.DM(B)
        # print(A)
        # print(B)
        return x_next_lin_fn


class Stack_model(Linear_model):
    def __init__(self, sys_fn, delta_t, N, x_init, C, D, E, cont_time=True, nonlinear=False, u_0=None, xr=None, ur=None):
        '''

        x_next = Ak x_k + Bk u_k + Ck w_k
        y_k = D x_k + E w_k

        Args:
            A: discretized and linearized
            B: discretized and linearized
            C: Discrete time
            D: Discrete time
            E: Discrete time

        '''
        super().__init__(sys_fn, delta_t, cont_time, nonlinear)

        self.xr = xr
        self.ur = ur
        self.N = N

        self.C = C
        self.D = D
        self.E = E
        self.x_init = x_init

        if nonlinear == True:
            self.A = self.A_fn(x_init, u_0).full()
            self.B = self.B_fn(x_init, u_0).full()
            self.delta = self.delta_fn(x_init, u_0).full()
        else:
            self.A = self.A.full()
            self.B = self.B.full()
            self.delta = self.delta  # delta = 0
        # print(self.A,self.B)

        Nx = self.Nx
        Nu = self.Nu
        Nd = np.shape(C)[1]  # Dimension of disturbance
        Nr = np.shape(D)[0]  # Dimension of output

        self.Nd = Nd
        self.Nr = Nr

        # dimension of stacked matrices

        Nx_s = (N + 1) * Nx
        Nu_s = N * Nu
        Ny_s = N * Nr
        Nw_s = N * Nd

        self.Nx_s = Nx_s
        self.Nu_s = Nu_s
        self.Ny_s = Ny_s
        self.Nw_s = Nw_s

        # Ax = np.zeros([Nx_s, Nx])
        # Bx = np.zeros([Nx_s, Nu_s])
        # Cx = np.zeros([Nx_s, Nw_s])
        # Ay = np.zeros([Ny_s, Nx])
        # By = np.zeros([Ny_s, Nu_s])
        # Cy = np.zeros([Ny_s, Nw_s])
        # Ey = np.zeros([Ny_s, Nw_s])

        self.stack_system()

    def stack_system(self):
        '''
        Stack system matrix for N prediction horizon

        x_next = A x_k + B u_k + C w_k
        y_k = D x_k + E w_k

        '''
        Nx = self.Nx  # Dimension of state
        Nu = self.Nu  # Dimension of input
        Nd = self.Nd  # Dimension of disturbance
        Nr = self.Nr  # Dimension of output

        Nx_s = self.Nx_s
        Nu_s = self.Nu_s
        Ny_s = self.Ny_s
        Nw_s = self.Nw_s

        N = self.N

        Ax = np.zeros([Nx_s, Nx])
        Bx = np.zeros([Nx_s, Nu_s])
        Cx = np.zeros([Nx_s, Nw_s])
        A_ext = np.zeros([Nx_s, Nx])
        Ay = np.zeros([Ny_s, Nx])
        By = np.zeros([Ny_s, Nu_s])
        Cy = np.zeros([Ny_s, Nw_s])
        Ey = np.zeros([Ny_s, Nw_s])
        Cx_tilde = np.zeros([Nx_s, Nw_s + 1])
        Cx_tilde_obj = np.zeros([Nx_s, Nw_s + 1])
        Cy_tilde = np.zeros([Ny_s + 1, Nw_s + 1])
        Ey_tilde = np.zeros([Ny_s + 1, Nw_s + 1])
        D_tilde = np.zeros([Ny_s + 1, Nx_s])
        #     H

        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E



        x_init = self.x_init

        # Ax
        for i in range(N + 1):
            Ax[i * Nx:(i + 1) * Nx, :] = matrix_power(A, i)
        # Bx
        for i in range(N):
            mat_temp = B
            for j in range(i+1):
                Bx[(i + 1) * Nx: (i + 2) * Nx, (i - j) * Nu: (i - j + 1) * Nu] = mat_temp #could be problematic
                mat_temp = A @ mat_temp
        # Cx
        for i in range(N):
            mat_temp = C
            for j in range(i+1):
                Cx[(i + 1) * Nx: (i + 2) * Nx, (i - j) * Nd: (i - j + 1) * Nd] = mat_temp
                mat_temp = A @ mat_temp

        # A_ext
        temp = 0
        for i in range(N + 1):
            if i == 0:
                temp = np.zeros(np.shape(A))
            else:
                temp += matrix_power(A, i - 1)
            A_ext[i * Nx:(i + 1) * Nx, :] = temp

        # Ay
        for i in range(N):
            Ay[i * Nr:(i + 1) * Nr, :] = D @ matrix_power(A, i)
        # # By
        # for i in range(N):
        #     mat_temp = B
        #     for j in range(i + 1):
        #         By[(i + 1) * Nr: (i + 2) * Nr, (i - j) * Nu: (i - j + 1) * Nu] = D @ mat_temp
        #         mat_temp = A @ mat_temp
        # Cy
        for i in range(N-1):
            mat_temp = C
            for j in range(i+1):
                Cy[(i + 1) * Nr: (i + 2) * Nr, (i - j) * Nd: (i - j + 1) * Nd] = D @ mat_temp
                # Cy[(i + 1) * Nr: (i + 2) * Nr, j * Nd: (j + 1) * Nd] = D @ matrix_power(A, j)@ C #here could be problematic
                mat_temp = A @ mat_temp
        # Ey
        for i in range(N):
            Ey[i * Nr: (i + 1) * Nr, i * Nd: (i + 1) * Nd] = E
        # Cx_tilde
        delta = self.delta
        if self.xr is not None:
            xr = self.xr
            xr_ext = np.tile(xr, (N + 1, 1))
        else:
            xr = np.zeros([Nx, 1])
            xr_ext = np.tile(xr, (N + 1, 1))

        # ur_ext
        if self.ur is not None:
            ur = self.ur
            ur_ext = np.tile(ur, (N, 1))
        else:
            ur = np.zeros([Nu, 1])
            ur_ext = np.tile(ur, (N, 1))


        #         print(Ax @ x_init)
        #         print(Ax @ x_init)
        Cx_tilde[:, [0]] = Ax @ x_init + A_ext @ delta
        Cx_tilde[:, 1:] = Cx


        # Cx_tilde_obj[:, [0]] = Ax @ x_init + A_ext @ delta - xr_ext - Bx @ ur_ext
        Cx_tilde_obj[:, [0]] = Ax @ x_init + A_ext @ delta - xr_ext
        # Cx_tilde_obj[:, [0]] = Ax @ (x_init - xr) + A_ext @ delta
        Cx_tilde_obj[:, 1:] = Cx

        # print(Cx_tilde_obj)
        # Cy_tilde
        # Cy_tilde[Nr:, [0]] = Ay @ x_init
        # Cy_tilde[Nr:, [0]] = Ay @ (x_init - xr)
        # Cy_tilde[Nr:, 1:] = Cy
        Cy_tilde[1:, [0]] = Ay @ (x_init - xr)
        Cy_tilde[1:, 1:] = Cy
        # Ey_tilde
        Ey_tilde[0, 0] = 1
        Ey_tilde[1:, 1:] = Ey


        self.xr_ext = xr_ext
        self.ur_ext = ur_ext

        self.Ax = Ax
        self.Bx = Bx
        self.Cx = Cx
        self.A_ext = A_ext
        self.Ay = Ay
        self.By = By
        self.Cy = Cy
        self.Ey = Ey
        self.Cx_tilde = Cx_tilde
        self.Cx_tilde_obj = Cx_tilde_obj
        self.Cy_tilde = Cy_tilde
        self.Ey_tilde = Ey_tilde



class Opt_problem(Stack_model):
    def __init__(self, sys_fn, delta_t, N, x_init, C, D, E, Q, Qf, R, cont_time=True, nonlinear=True, u_0=None, xr=None,
                 ur=None, collect=False, est=False, mu=None, sigma=None, beta=1, sin_const=1, N_sample=1, epsilon=1,
                 i_th_state=1, i_state_ub=0.5):
        super().__init__(sys_fn, delta_t, N, x_init, C, D, E, cont_time, nonlinear, u_0, xr, ur)

        N = self.N

        Nx = self.Nx  # Dimension of state
        Nu = self.Nu  # Dimension of input
        Nd = self.Nd  # Dimension of disturbance
        Nr = self.Nr  # Dimension of output

        Nx_s = self.Nx_s
        Nu_s = self.Nu_s
        Ny_s = self.Ny_s
        Nw_s = self.Nw_s
        # Stack system matrices
        Ax = self.Ax
        Bx = self.Bx
        Cx = self.Cx
        A_ext = self.A_ext
        Ay = self.Ay
        By = self.By
        Cy = self.Cy
        Ey = self.Ey
        Cx_tilde = self.Cx_tilde
        Cx_tilde_obj = self.Cx_tilde_obj
        Cy_tilde = self.Cy_tilde
        Ey_tilde = self.Ey_tilde

        # print("delta", self.delta)
        # print("xr_ext", self.xr_ext)
        # print("A",self.A)
        # print("B",self.B)
        # print("C",self.C)
        # print("D",self.D)
        # print("E",self.E)
        # print("Ax", self.Ax)
        # print("Bx", self.Bx)
        # print("Cx", self.Cx)
        # print("A_ext", self.A_ext)
        # print("Ay", self.Ay)
        # print("By", self.By)
        # print("Ey", self.Ey)
        # print("Cx_tilde", self.Cx_tilde)
        # print("Cy_tilde", self.Cy_tilde)
        # print("Ey_tilde", self.Ey_tilde)

        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.sin_const = sin_const
        self.N_sample = N_sample
        self.epsilon = epsilon
        self.beta = beta
        if not est:
            if mu is not None:
                self.mu = mu
            else:
                print("mu not given")
            if sigma is not None:
                self.sigma = sigma
            else:
                print("sigma not given")
        #         else
        ## Estimate either using offline data or update algorithm using online data
        #             mu_w, M_w = self.mean_covariance(N, d, data_set=data_set)

        #     mu_w, M_w = self.mean_covariance(N, d, data_set=None)
        # if est:
        #     mu_w, M_w = self.mean_covariance(N, d, data_set=data_set)
        #
        # self.mu_w = mu_w
        # self.M_w = M_w

        # Generate disturbance data at each sampling time
        if not collect:
            W_sample_matrix, W_sample_matrix_ext = self.gene_disturbance(N_sample, sin_const)
            self.W_sample_matrix = W_sample_matrix
            self.W_sample_matrix_ext = W_sample_matrix_ext
            data_set = W_sample_matrix
            self.mean_covariance(data_set=data_set, est=est)
            # print(W_sample_matrix)
            self.define_loss_func(Q, Qf, R, sin_const)
            # W_sample_matrix, W_sample_matrix_ext = self.disturbance_para(N, d, N_sample, sin_const)
            self.define_constraint(i_th_state, i_state_ub)

        # select data from the disturbance data pool and extend the pool at each sampling time
        # else:
        #     W_sample_matrix, W_sample_matrix_ext = self.select_disturbance(N, d, N_sample, sin_const, data_set)
        #     self.W_sample_matrix = W_sample_matrix
        #     self.W_sample_matrix_ext = W_sample_matrix_ext
        #
        #     self.mean_covariance(data_set=data_set)
        #     self.define_loss_func(Q, Qf, R, sin_const)
        #     self.define_constraint(i_th_state, i_state_ub)

        loss_func = self.loss_func
        constraint = self.constraint
        self.obj = cp.Minimize(loss_func)
        self.prob = cp.Problem(self.obj, constraint)

    # def select_disturbance(self, sin_const, data_set):
    #     N = self.N
    #     d = self.d
    #
    #     W_sample_matrix = np.vstack(data_set[- N * N_sample:])
    #     W_sample_matrix = W_sample_matrix.T.reshape(d * N, -1, order='F')
    #     W_sample_matrix_ext = np.vstack([np.ones([1, N_sample]), W_sample_matrix])
    #
    #     return W_sample_matrix, W_sample_matrix_ext

    def gene_disturbance(self, N_sample, sin_const):
        # Generate data: const * sinx

        N = self.N
        Nd = self.Nd
        w_sample = []
        for i in range(N_sample):
            w_temp = sin_const * np.sin(np.random.randn(N * Nd))
            w_sample += [w_temp]
        W_sample_matrix = np.array(w_sample).T

        W_sample_matrix_ext = np.vstack([np.ones([1, N_sample]), W_sample_matrix])
        return W_sample_matrix, W_sample_matrix_ext

    # def disturbance_para(self, N, d, N_sample, sin_const):
    #     W_sample_matrix = cp.Parameter((N * d, N_sample))
    #     W_sample_matrix_ext = cp.vstack([np.ones([1, N_sample]),W_sample_matrix])
    #     return W_sample_matrix, W_sample_matrix_ext

    def mean_covariance(self, data_set=None, est=False):

        N = self.N
        Nd = self.Nd

        if est is False:
            mu = self.mu
            sigma = self.sigma
            sin_const = self.sin_const
            mu_w = np.vstack([1] + [mu] * N)
            #     M_w = mu_w @ mu_w.T + np.diag([0] + [1] * N * d)
            # M_w = np.diag([0.000001] + [sin_const ** 2 * (1 - np.exp(-2 * sigma ** 2)) / 2] * N * Nd)
            M_w = np.diag([0] + [sin_const ** 2 * (1 - np.exp(-2 * sigma ** 2)) / 2] * N * Nd)
        # elif est is True:
        #     if data_set is None:
        #         print("mean_covariance function error")
        #     else:
        #         # Estimate mean and covariance from data
        #         # print("data",data_set)
        #         if isinstance(data_set, list):
        #             W_sample_matrix = np.vstack(data_set)
        #         else:
        #             W_sample_matrix = data_set
        #         W_sample_matrix = W_sample_matrix.T.reshape(Nd, -1, order='F')
        #         # print(W_sample_matrix)
        #         est_mean = np.mean(W_sample_matrix, axis=1).reshape(-1, 1)
        #         est_var = np.var(W_sample_matrix, axis=1, ddof=1).flatten().tolist()
        #
        #         mu_w = np.vstack([1] + [est_mean] * N)
        #         M_w = np.diag([1] + est_var * N)
        # print("M_w", M_w)
        self.mu_w = mu_w
        self.M_w = M_w
        # return mu_w, M_w

    def define_loss_func(self, Q, Qf, R, sin_const):
        N = self.N

        Nx = self.Nx  # Dimension of state
        Nu = self.Nu  # Dimension of input
        Nd = self.Nd  # Dimension of disturbance
        Nr = self.Nr  # Dimension of output

        Nx_s = self.Nx_s
        Nu_s = self.Nu_s
        Ny_s = self.Ny_s
        Nw_s = self.Nw_s

        Bx = self.Bx
        # Cx_tilde = self.Cx_tilde
        Cx_tilde = self.Cx_tilde_obj
        beta = self.beta


        # Define decision variables for POB affine constrol law
        H_cal_dec = cp.Variable((Nu_s, 1))
        #     print(H_cal_dec)
        # \mathcal{H} matrix
        for i in range(N):
            H_col = cp.Variable((Nu_s - (i * Nu), Nr))
            if i > 0:
                H_col = cp.vstack([np.zeros([i * Nu, Nr]), H_col])
            #         print(H_col)
            H_cal_dec = cp.hstack([H_cal_dec, H_col])
        #     print(H_cal_dec)
        #     print(np.shape(H_cal_dec))
        # Define intermediate decision variables for objective function
        H = cp.Variable((Nu_s, Nw_s + 1))

        # Define loss function
        #     beta = 0.95
        Jx = np.zeros([(N + 1) * Nx, (N + 1) * Nx])
        for i in range(N):
            Jx[i * Nx: (i + 1) * Nx, i * Nx: (i + 1) * Nx] = beta ** i * Q
        Jx[N * Nx:, N * Nx:] = beta ** N * Qf

        Ju = np.zeros([N * Nu, N * Nu])
        for i in range(N):
            Ju[i * Nu: (i + 1) * Nu, i * Nu: (i + 1) * Nu] = beta ** i * R


        # This is only for var = 1 and mean = 0. Should be modified.
        mu_w = self.mu_w
        M_w = self.M_w
        # print("M_w value", M_w)
        # mu_w = np.vstack([1] + [mu] * N)
        #     M_w = mu_w @ mu_w.T + np.diag([0] + [1] * N * d)
        # M_w = np.diag([1] + [sin_const ** 2 * (1 - np.exp(-2 * sigma ** 2)) / 2] * N * d)

        # Intermediate decision variables. Since CVXPY does not support quadratic obj of decision variable matrix.
        H_new_matrix = []
        #     for i in range(Nw+1):
        #         H_new_matrix += [cp.Variable([Nu,1])]
        #     H_new = cp.hstack(H_new_matrix)
        for i in range(Nu_s):
            H_new_matrix += [cp.Variable((1, Nw_s + 1))]
        H_new = cp.vstack(H_new_matrix)

        #     print(H_new.shape)
        # Reformulate the quadratic term
        # Ju = np.zeros(np.shape(Ju))
        eigval, eigvec = np.linalg.eig(Ju + Bx.T @ Jx @ Bx)
        eigval_mat = np.diag(eigval)
        # print("EV,EM",eigval,eigvec)
        # print("Eigen value", Ju + Bx.T @ Jx @ Bx - eigvec @ eigval_mat @ np.linalg.inv(eigvec))
        # Loss function
        loss_func = 0

        N_eig = np.shape(eigval)[0]
        # I = np.diag([1] * (Nw_s + 1))
        for i in range(N_eig):
            # Reformulate Tr[(H.T @ (Ju + Bx.T @ Jx @ Bx)@ H ) @ M_w ]
            #         print(np.shape(H_new_matrix[i].T))
            # loss_func += eigval[i] * M_w[i, i] * cp.quad_form(H_new_matrix[i].T, I)  # When M_w is identity matrix. Otherwise reformulate system matrix or this line
            # print(M_w)
            # loss_func += eigval[i] * M_w[i, i] * cp.quad_form(H_new_moatrix[i].T, I)  # When M_w is identity matrix. Otherwise reformulate system matrix or this line
            # if np.trace(M_w) == 0:
            #     continue
            loss_func += eigval[i] * cp.quad_form(H_new_matrix[i].T, M_w)  # When M_w is identity matrix. Otherwise reformulate system matrix or this line
            # for j in range(np.shape(H_new_matrix[i])[1]):
                # loss_func += eigval[i] * cp.trace( H_new_matrix[i].T @ H_new_matrix[i] @ M_w[j,j]) # When M_w is identity matrix. Otherwise reformulate system matrix or this line
                # print(np.shape(H_new[i,j]))
                # loss_func += eigval[i] * cp.quad_form(H_new[i,j],1) * M_w[j, j]  # When M_w is identity matrix. Otherwise reformulate system matrix or this line
        #     loss_func += cp.trace(2 * Cx_tilde.T @ Jx @ Bx @ eigvec @ H_new  @ M_w)
        loss_func += cp.trace(2 * Cx_tilde.T @ Jx @ Bx @ H @ M_w)
        #     loss_func += cp.trace(2 * Cx_tilde.T @ Jx @ Bx @ H_cal_dec @ (Cy_tilde + Ey_tilde) @ M_w)
        loss_func += cp.trace(Cx_tilde.T @ Jx @ Cx_tilde @ M_w)
        # Reformulate mu_w.T @ (H.T @ (Ju + Bx.T @ Jx @ Bx)@ H ) @ mu_w
        #     loss_func += eigval[0] * cp.quad_form(H_new_matrix[0].T, I) +  2 * mu_w.T @  Cx_tilde.T @ Jx @ Bx @ eigvec @ H_new @ mu_w
        # loss_func += eigval[0] * cp.quad_form(H_new_matrix[0].T, I) + 2 * mu_w.T @ Cx_tilde.T @ Jx @ Bx @ H @ mu_w
        # loss_func += eigval[0] * cp.quad_form(H_new_matrix[0].T, I)
        loss_func += cp.quad_form(H_new[:,0], eigval_mat)
        # print(np.shape(mu_w),np.shape(Cx_tilde), np.shape(Jx), np.shape(Bx), np.shape(H))
        loss_func += 2 * mu_w.T @ Cx_tilde.T @ Jx @ Bx @ H @ mu_w
        #     loss_func += eigval[0] * cp.quad_form(H_new_matrix[0].T, I) +  2 * mu_w.T @  Cx_tilde.T @ Jx @ Bx @ H_cal_dec @ (Cy_tilde + Ey_tilde) @ mu_w
        loss_func += mu_w.T @ Cx_tilde.T @ Jx @ Cx_tilde @ mu_w
        # Reference for input. If no input ref is given, then ur_ext is 0.
        ur_ext = self.ur_ext
        xr_ext = self.xr_ext
        loss_func += -2 * ur_ext.T @ Ju @ H @ mu_w
        loss_func += ur_ext.T @ Ju @ ur_ext

        # loss_func += -2 * xr_ext.T @ Jx @ (Cx_tilde + Bx @ H) @ mu_w
        #         print(loss_func)

        self.Jx = Jx
        self.Ju = Ju
        self.eigval = eigval
        self.eigvec = eigvec
        self.H_cal_dec = H_cal_dec
        self.H = H
        self.H_new_matrix = H_new_matrix
        self.H_new = H_new
        self.loss_func = loss_func

    def define_constraint(self, i_th_state, i_state_ub):

        W_sample_matrix = self.W_sample_matrix
        W_sample_matrix_ext = self.W_sample_matrix_ext
        eigval = self.eigval
        eigvec = self.eigvec
        H_cal_dec = self.H_cal_dec
        H = self.H
        H_new = self.H_new

        N = self.N
        N_sample = self.N_sample

        Nx = self.Nx  # Dimension of state
        Nu = self.Nu  # Dimension of input
        Nd = self.Nd  # Dimension of disturbance
        Nr = self.Nr  # Dimension of output

        Nx_s = self.Nx_s
        Nu_s = self.Nu_s
        Ny_s = self.Ny_s
        Nw_s = self.Nw_s

        Bx = self.Bx
        Cx_tilde = self.Cx_tilde
        Cy_tilde = self.Cy_tilde
        Ey_tilde = self.Ey_tilde

        sin_const = self.sin_const
        epsilon = self.epsilon

        constraint = []
        constraint += [H_new == np.linalg.inv(eigvec) @ H]
        # constraint += [eigvec @ H_new == H]
        #             constraint += [ H_new == eigvec.T @ H ]
        constraint += [H == H_cal_dec @ (Cy_tilde + Ey_tilde)]
        # constraint += [H_new == np.linalg.inv(eigvec) @ H_cal_dec @ (Cy_tilde + Ey_tilde) ]

        #     i_th_state = 1 # 0 for first element, 1 for second element
        #     i_state_ub = 0.05

        d_supp = np.vstack((sin_const * np.ones([N * Nd, 1]), sin_const * np.ones([N * Nd, 1])))
        C_supp = np.vstack((np.diag([1] * N * Nd), np.diag([-1] * N * Nd)))
        #     d_supp = np.vstack( ( 0 * np.ones([N*d, 1]), 0 * np.ones([N*d, 1])))
        #     C_supp = np.vstack( (np.diag([0]*N*d), np.diag([0]*N*d) ))
        #     lambda_var = cp.Variable()
        lambda_var = cp.Variable(nonneg=True)

        gamma_shape = np.shape(d_supp)[0]
        gamma_matrix = []
        for i in range(N_sample):
            for j in range(N):
                gamma_var = cp.Variable((gamma_shape, 1), nonneg=True)
                #             gamma_var = cp.Variable([gamma_shape,1])
                gamma_matrix += [gamma_var]
        # k in N, i in N_sample
        # bk + <ak,xi_i>
        X_constraint = (Bx @ H_cal_dec @ (Cy_tilde + Ey_tilde) + Cx_tilde) @ W_sample_matrix_ext
        #     si_var = cp.Variable((N_sample,1))
        si_var = cp.Variable(N_sample)

        for i in range(N_sample):
            for j in range(N):
                #             print(N_sample)
                constraint_temp = X_constraint[Nx * (j + 1) + i_th_state, i] + gamma_matrix[i * N + j].T @ (
                        d_supp - C_supp @ W_sample_matrix[:, [i]])
                #             constraint += [constraint_temp <= si_var[i,0]]
                constraint += [constraint_temp <= si_var[i]]
        # print("constraint_temp", constraint_temp.shape)
        ak_matrix = (Bx @ H_cal_dec @ (Cy_tilde + Ey_tilde) + Cx_tilde)[:, 1:]
        for i in range(N_sample):
            for j in range(N):
                #             constraint_temp = C_supp.T @ gamma_matrix[i * N + j] - ak_matrix[n * (j+1) + i_th_state:n * (j+1)+i_th_state + 1,:].T
                constraint_temp = C_supp.T @ gamma_matrix[i * N + j] - ak_matrix[[Nx * (j + 1) + i_th_state], :].T
                #             constraint += [cp.norm_inf(constraint_temp) <= lambda_var]
                constraint += [cp.norm(constraint_temp, p=np.inf) <= lambda_var]

        #     for i in range(N_sample):
        #         for j in range(N):
        #             constraint += [gamma_matrix[i * N + j] >= 0]
        #     constraint += [lambda_var * epsilon + 1/N_sample * cp.sum(si_var) <= i_state_ub]
        constraint += [lambda_var * epsilon + 1 / N_sample * cp.sum(si_var) <= i_state_ub]

        self.lambda_var = lambda_var
        self.gamma_matrix = gamma_matrix
        self.si_var = si_var
        self.constraint = constraint


#     def define_new_opt(self, x_init, xr, ur, u_0 = None):
#         self.restack_system(x_init, xr = xr, ur = ur, u_0 = u_0)

class Simulation():
    '''
    @Arg

        mode: "collect" data and incorporate the collected data into constraint
              "gene" data at each time instant and use fixed number of data to solve opt problem

    '''

    def __init__(self, sys_fn, delta_t, N, x_init, C, D, E, Q, Qf, R, cont_time=True, nonlinear=True, u_0=None,
                 xr=None, ur=None, collect=False, est=False, mu=None, sigma=None, beta=1,
                 sin_const=1, N_sample=5, epsilon=1, i_th_state=1, i_state_ub=0.5, N_sim=80,
                 mode="gene", data_set=None, N_sample_max=None):
        #         super().__init__(sys_fn, delta_t, N, x_init, C, D, E, Q, Qf, R, cont_time, nonlinear, u_0, xr, ur, est, mu, sigma, beta, sin_const, N_sample, epsilon, i_th_state, i_state_ub)

        self.sys_fn = sys_fn
        self.delta_t = delta_t
        self.N = N
        self.x_init = x_init
        self.C = C
        self.D = D
        self.E = E
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.cont_time = cont_time
        self.nonlinear = nonlinear
        self.u_0 = u_0
        self.xr = xr
        self.ur = ur
        self.collect = collect
        self.est = est
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.sin_const = sin_const
        self.N_sample = N_sample
        self.epsilon = epsilon
        self.i_th_state = i_th_state
        self.i_state_ub = i_state_ub
        self.N_sim = N_sim
        self.mode = mode
        self.data_set = data_set
        self.N_sample_max = N_sample_max

#         opt_problem = Opt_problem(sys_fn, delta_t, N, x_init, C, D, E, Q, Qf, R, cont_time=cont_time, nonlinear=nonlinear, u_0=u_0,
#                                   xr=xr, ur=ur, collect=collect, est=est, mu=mu, sigma=sigma, beta=beta, sin_const=sin_const,
#                                   N_sample=N_sample, epsilon=epsilon, i_th_state=i_th_state, i_state_ub=i_state_ub)

        self.x_sim, self.u_sim = self.simulation_gene(x_init, N_sim)

        # TODO: finish coding collect



    def simulation_gene(self, x_init, N_sim):

        sys_fn = self.sys_fn
        delta_t = self.delta_t
        N = self.N
        x_init = self.x_init
        C = self.C
        D = self.D
        E = self.E
        Q = self.Q
        Qf = self.Qf
        R = self.R
        cont_time = self.cont_time
        nonlinear = self.nonlinear
        u_0 = self.u_0
        xr = self.xr
        ur = self.ur
        collect = self.collect
        est = self.est
        mu = self.mu
        sigma = self.sigma
        beta = self.beta
        sin_const = self.sin_const
        N_sample = self.N_sample
        epsilon = self.epsilon
        i_th_state = self.i_th_state
        i_state_ub = self.i_state_ub
        N_sim = self.N_sim
        mode = self.mode
        data_set = self.data_set
        N_sample_max = self.N_sample_max

        ode = sys_fn

        t0 = 0
        xk = x_init
        uk = u_0
        t = t0
        h = delta_t

        x_list = []
        x_list += xk.flatten().tolist()
        u_list = []

        xk = x_init
        for i in range(N_sim):
            #     if i % N == 0:
            # self.model.change_xinit(xk - np.array([[1],[0]]))
            #             self.model.change_xinit(xk)
            #             print(self.model.Ak)
#             print(xr,ur)

            # if i>= 75:
            #     i_state_ub = ca.inf # for nonlinear inverted pendulum
            opt_problem = Opt_problem(sys_fn, delta_t, N, xk, C, D, E, Q, Qf, R, cont_time=cont_time, nonlinear=nonlinear, u_0=uk,
                                      xr=xr, ur=ur, collect=collect, est=est, mu=mu, sigma=sigma, beta=beta, sin_const=sin_const,
                                      N_sample=N_sample, epsilon=epsilon, i_th_state=i_th_state, i_state_ub=i_state_ub)
            self.Nx = opt_problem.Nx
            Nd = opt_problem.Nd
            # W_sample, W_sample_ext = self.gene_disturbance(N, d, N_sample, sin_const)
            # opt_problem.W_sample_matrix.value = W_sample
            prob = opt_problem.prob
            #         print(W_sample_matrix)
            #     print( prob.solve(verbose=True))
            # prob.solve(solver=cp.MOSEK,verbose = True)
            try:
                # prob.solve(solver=cp.MOSEK)
                prob.solve(solver=cp.GUROBI)
                # prob.solve(solver=cp.MOSEK, verbose = True)
            except ValueError as e:
                print('error type: ', type(e))
                print("current state and input", xk, uk)
                # print("solver state",prob.status)
                # continue


            # print("opt value:", prob.value)
            #     print( prob.solve(verbose=True))
            #         prob.solve(solver = cp.MOSEK,verbose = True, mosek_params = {mosek.dparam.basis_tol_s:1e-9, mosek.dparam.ana_sol_infeas_tol:0})
            #         print(Ax @ x_init +  Bx @ H.value @ W_sample_matrix_ext[:,0:1]  + Cx @ W_sample_matrix[:,0:1])
            #         print("status:", prob.status)
            # print("Controller", opt_problem.H_cal_dec.value[0,0], opt_problem.H_cal_dec.value[0,1])
            #         print("dual:", constraint[0].dual_value)
            #         print("gamma", gamma_matrix[0].value,  gamma_matrix[1].value,  gamma_matrix[2].value,  gamma_matrix[3].value)
            # print("lambda",opt_problem.lambda_var.value)
            #         print("lambda time epsilon",lambda_var.value * epsilon)
            # print("si",opt_problem.si_var.value)
            # print("si average",np.sum(opt_problem.si_var.value)/N_sample)
            # print("state_constraint", np.mean(opt_problem.si_var.value) + opt_problem.lambda_var.value * epsilon)
            # print("state",(opt_problem.Bx @ opt_problem.H_cal_dec.value @ (opt_problem.Cy_tilde + opt_problem.Ey_tilde) + opt_problem.Cx_tilde) @ opt_problem.W_sample_matrix_ext)
            mu_w_temp = opt_problem.mu_w
            ur_ext_temp = opt_problem.ur_ext
            temp_v = (opt_problem.Bx @ opt_problem.H.value + opt_problem.Cx_tilde_obj) @ mu_w_temp
            # print("obj",temp_v.T @ opt_problem.Jx @ temp_v + opt_problem.H.value.T @ opt_problem.Ju @ opt_problem.H.value)
            # print("value 1", temp_v.T @ opt_problem.Jx @ temp_v)
            # print("value 2", (opt_problem.H.value @ mu_w_temp - ur_ext_temp).T @ opt_problem.Ju @ (opt_problem.H.value @ mu_w_temp - ur_ext_temp))

            # print("obj",temp_v.T @ opt_problem.Jx @ temp_v + (opt_problem.H.value @ mu_w_temp - ur_ext_temp).T @ opt_problem.Ju @ (opt_problem.H.value @ mu_w_temp - ur_ext_temp))
            # print("disturbance data", W_sample_matrix)

            wk = sin_const * np.sin(np.random.randn(Nd, 1))
            # uk = opt_problem.H_cal_dec.value[0, 0] + opt_problem.H_cal_dec.value[0, 1] * (D @ xk  + E @ wk)
            uk = opt_problem.H_cal_dec.value[0, 0] + opt_problem.H_cal_dec.value[0, 1] * (D @ (xk - xr) + E @ wk)

            # print(opt_problem.H_cal_dec.value[0, 0])
            # print(opt_problem.H_cal_dec.value[0, 1])
            # print(opt_problem.constraint)
            # print(-2 * opt_problem.ur_ext.T @ opt_problem.Ju @ H @ mu_w)
            # uk = opt_problem.H_cal_dec.value[0, 0] + opt_problem.H_cal_dec.value[0, 1] * (D @ (xk-np.array([[1],[0]]))+ E @ wk)
            # uk = uk + ur
            u_list += uk.flatten().tolist()
            # print("current state and input", xk, uk)
            # x_kp1 = self.simulation_Euler(Ak, Bk, xk, uk)
            x_kp1 = opt_problem.dt_sys_fn(xk, uk).full()
#             print(opt_problem.dt_sys_fn(xk, uk),type(opt_problem.dt_sys_fn(xk, uk)))
            # x_kp1 = Ak @ xk + Bk @ uk
            # x_kp1 = self.RK4_np(self.inverted_pendulum_ode, xk, uk, t, h)
            xk = x_kp1
            xk += C @ wk
            x_list += xk.flatten().tolist()
        return x_list, u_list

    def gene_disturbance(self, N, d, N_sample, sin_const):
        # Generate data: const * sinx

        w_sample = []
        for i in range(N_sample):
            w_temp = sin_const * np.sin(np.random.randn(N * d))
            w_sample += [w_temp]
        W_sample_matrix = np.array(w_sample).T

        W_sample_matrix_ext = np.vstack([np.ones([1, N_sample]), W_sample_matrix])
        return W_sample_matrix, W_sample_matrix_ext



    def plot_state(self):
        delta_t = self.delta_t
        Nx = self.Nx

        x_traj = self.x_sim

        Nt = np.shape(x_traj[::Nx])[0]
        t_plot = [delta_t * i for i in range(Nt)]

        plt.figure(1, figsize=(10, 20))
        plt.clf()
        # Print states
        for i in range(Nx):
            plt.subplot(str(Nx) + str(1) + str(i + 1))
            plt.grid()
            x_traj_temp = x_traj[i::Nx]
            plt.plot(t_plot, x_traj_temp)
            plt.ylabel('x' + str(i + 1))

            # Print reference
            ref_plot_temp = [ self.xr[i] ] * Nt
            plt.plot(t_plot,ref_plot_temp,color = "k")

            # Print constraint
            if i == self.i_th_state:
                v_constr = self.i_state_ub
                constr_plot_temp = [v_constr] * Nt
                plt.plot(t_plot, constr_plot_temp, color="r")



        plt.xlabel('t')
        plt.show()
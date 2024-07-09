import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate


class Model:
    """
    Solves a optimal control problem:
    x(t) = f(x(t), u(t), u_discrete, du, x0)
    x_min <= x <= xmax for all t
    u_min <= u <= u_max
    g_min <= g(x, u) <= g_max
    g_end_min <= g_end <= g_end_max
    x0 = x(0)
    J = M(x, t_end) + int(L(x, u, t_end) dt)

    set limits to np.inf or -np.inf is non apply
    one of M or L can be set to None
    set g_min = g_max = 0 for equality constraints
    """

    def __init__(
        self,
        f,  # rigth side of the differential equation (f(x, u, du, x0))
        #  biomass has to be the first variable to calculate n correctly
        x0_min,
        x0_max,  # initial values
        x_min,  # minimal values for x in all timepoints, sets the dim of x
        x_max,  # maximal values for x
        u_min,  # minimal vlaue for the control
        u_max,
        u0=None,
        u0_max=None,
        u_discrete_min=[],
        u_discrete_max=[],
        g=None,  # list of functions, constraints valid for all timepoints g(x, u, du)
        g_min=None,
        g_max=None,
        g_end=None,  # list of functions (g_end(x_end))
        g_end_min=None,
        g_end_max=None,
        g_x0=None,
        g_x0_min=None,
        g_x0_max=None,
        M=None,  # function M(x_end, t_end)
        L=None,  # function L(x, u)
        N=10,  # number of intervals for discretisation
        p_order=2,  # order of polynom used for descretisation
        t_min=0,  # minimal process time
        t_max=50,  # maximal process time
        solver_options={
            # "ipopt.tol": 1e-10,
            # "ipopt.constr_viol_tol": 1e-14,
        },  # ipopt options
        x_names=None,  # name of variables
        u_names=None,  # name of controls
        u_guess=None,  # initial guess for the control as function u_guess(t)
        du_guess=None,
        end_guess=[],  # f(t,x): creterium to end initial guess before t_max, emptly list or list of functions
        # integration for initial guess terminates if end_guess is zero
        ode_solver_options={
            "method": "Radau"
        },  # options for the ode solver used to get initial values
        collocation_polynom="legendre",  # used collocation polynoms ("legendre" or "radau")
        du_penalty_factor=None,  # add an penalty factor for fast control changes ((u[i+1] - u[1]) / (u[i+1] + u[i]))**2
        ddu_penaltiy_factor=None,  # penaltilizes changes in derivation of control
    ):
        self.f = f
        self.x0_min = x0_min
        self.x0_max = x0_max
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        self.u0_min = u0
        if u0_max is None:
            self.u0_max = u0
        else:
            self.u0_max = u0_max
        self.u_discrete_min = u_discrete_min
        self.u_discrete_max = u_discrete_max
        self.u_discrete_guess = [
            (x + y) / 2 for x, y in zip(self.u_discrete_min, self.u_discrete_max)
        ]
        self.g = g
        self.g_min = g_min
        self.g_max = g_max
        self.g_end = g_end
        self.g_end_min = g_end_min
        self.g_end_max = g_end_max
        self.g_x0 = g_x0
        self.g_x0_min = g_x0_min
        self.g_x0_max = g_x0_max
        self.M = M
        self.L = L
        self.N = N
        self.p_order = p_order
        self.t_min = t_min
        self.t_max = t_max
        self.solver_options = solver_options
        self.x_names = x_names
        self.u_names = u_names
        self.u_guess = u_guess
        self.du_guess = du_guess
        self.end_guess = end_guess
        self.ode_solver_options = ode_solver_options
        self.collocation_polynom = collocation_polynom

        self.x_dim = len(self.x_min)
        self.u_dim = len(self.u_max)
        if du_penalty_factor is None:
            self.du_penalty_factor = [0] * self.u_dim
        else:
            self.du_penalty_factor = du_penalty_factor
        if ddu_penaltiy_factor is None:
            self.ddu_penalty_factor = [0] * self.u_dim
        else:
            self.ddu_penalty_factor = ddu_penaltiy_factor
        # set globally used variables and functions

        self.x = cas.SX.sym("x", self.x_dim)
        self.u = cas.SX.sym("u", self.u_dim)
        self.du = cas.SX.sym("du", self.u_dim)
        self.x0 = cas.SX.sym("X0", self.x_dim)
        self.t_end = cas.SX.sym(
            "t_end"
        )  # scales the equation form [0, 1] to final time interval
        self.u_discrete = cas.SX.sym("u_discrete", len(self.u_discrete_min))
        self.f_cas = cas.Function(
            "f_cas",
            [self.x, self.u, self.u_discrete, self.du, self.t_end, self.x0],
            [
                cas.vertcat(*self.f(self.x, self.u, self.u_discrete, self.du, self.x0))
                * self.t_end
            ],
        )

    def solve(self):
        t_values = np.linspace(0, 1, self.N + 1)

        # i: interval
        # j: collocation point
        # k: equation dim
        # l: control dim

        # create discrete variables
        # x_points are at the end of intervals
        # x_col are the x on the collocation points
        # u_points are the constraints
        x_points = [
            [cas.SX.sym(f"x_points_{i}_{k}") for k in range(self.x_dim)]
            for i in range(self.N + 1)
        ]

        x_col = [
            [
                [cas.SX.sym(f"x_col_{i}_{j}_{k}") for k in range(self.x_dim)]
                for j in range(self.p_order)
            ]
            for i in range(self.N)
        ]

        u_points = [
            [cas.SX.sym(f"u_points_{i}_{l}") for l in range(self.u_dim)]
            for i in range(self.N + 1)
        ]

        u_discrete = [
            cas.SX.sym(f"u_discrete{i}") for i in range(len(self.u_discrete_min))
        ]

        # create a list with all x_points, x_col, u and t_end
        x_all = [b for a in x_points for b in a]
        x_all += [c for a in x_col for b in a for c in b]
        x_all += [b for a in u_points for b in a]
        x_all += u_discrete
        x_all.append(self.t_end)

        # set lower and upper bounds
        xlb = self.x0_min + self.x_min * ((self.p_order + 1) * self.N)
        xub = self.x0_max + self.x_max * ((self.p_order + 1) * self.N)
        ulb = self.u0_min + [self.u_min] * (self.N)
        uub = self.u0_max + [self.u_max] * (self.N)

        lbx = xlb + ulb + self.u_discrete_min + [self.t_min]
        ubx = xub + uub + self.u_discrete_max + [self.t_max]

        # create constraints for ODEs

        g = list()

        col_points = cas.collocation_points(self.p_order, self.collocation_polynom)
        C, D, B = cas.collocation_coeff(col_points)

        # create f[i][j][k] = f(x_i_j, u_i_j))[k], u_i_j is the linear itepolation between u_i and u_i+1
        f = list()
        for i in range(self.N):
            fi = list()
            du = cas.vertcat(
                *[
                    (u_points[i + 1][l] - u_points[i][l])
                    / ((t_values[i + 1] - t_values[i]) * self.t_end)
                    for l in range(self.u_dim)
                ]
            )
            for j in range(self.p_order):
                x_i_j = cas.vertcat(*x_col[i][j])
                u_i_j = cas.vertcat(*u_points[i]) * col_points[j] + cas.vertcat(
                    *u_points[i + 1]
                ) * (1 - col_points[j])
                f_i_j = self.f_cas(
                    x_i_j,
                    u_i_j,
                    cas.vertcat(*u_discrete),
                    du,
                    self.t_end,
                    cas.vertcat(*x_points[0]),
                )
                fi.append(cas.vertsplit(f_i_j))
            f.append(fi)

        for i in range(self.N):
            for k in range(self.x_dim):
                # continous equation:
                Z = cas.vertcat(
                    x_points[i][k], *[x_col[i][j][k] for j in range(self.p_order)]
                )
                g.append(Z.T @ D - x_points[i + 1][k])

                # derivates dp = f
                dp = Z.T @ C * self.N
                g += [dp[j] - f[i][j][k] for j in range(self.p_order)]

        glb = [0] * len(g)
        gub = [0] * len(g)

        # set optimization target J
        J = 0
        if self.M is not None:
            M_cas = cas.Function(
                "M_cas", [self.x, self.t_end], [self.M(self.x, self.t_end)]
            )
            x_end = cas.vertcat(*x_points[self.N])
            J += M_cas(x_end, self.t_end)

        if self.L is not None:
            L = 0
            for i in range(self.N):
                L_cas = cas.Function(
                    "L_cas", [self.x, self.u], [self.L(self.x, self.u)]
                )
                for j in range(self.p_order):
                    x_local = cas.vertcat(*[x_col[i][j][k] for k in range(self.x_dim)])
                    u_local = u_points[i][j] * col_points[j] + u_points[i + 1][j] * (
                        1 - col_points[j]
                    )
                    L += L_cas(x_local, u_local) * B[j]
            L *= self.t_end / self.N
            J += L

        # penaltiese for quick changes in control
        for l in range(self.u_dim):
            if self.du_penalty_factor[l] != 0:
                J_step = 0
                for i in range(1, self.N):
                    J_step += (
                        (u_points[i][l] - u_points[i + 1][l])
                        / (u_points[i][l] + u_points[i + 1][l])
                    ) ** 2
                J += J_step * self.du_penalty_factor[l] / self.N

        # penalties for quick changes of derivates of control
        # 0 if there is no change in derivate

        for l in range(self.u_dim):
            if self.ddu_penalty_factor[l] != 0:
                J_step = 0
                for i in range(2, self.N):
                    J_step += (
                        (u_points[i][l] - u_points[i - 1][l]) ** 2
                        + (u_points[i + 1][l] - u_points[i][l]) ** 2
                        - 4 * (u_points[i + 1][l] - u_points[i + 1][l]) ** 2
                    ) / self.N**2
                J += J_step * self.ddu_penalty_factor[l]

        # add user supplied constraints

        if self.g is not None:
            for fg in self.g:
                for i in range(self.N):
                    du = cas.vertcat(
                        *[
                            (u_points[i + 1][l] - u_points[i][l])
                            / ((t_values[i + 1] - t_values[i]) * self.t_end)
                            for l in range(self.u_dim)
                        ]
                    )
                    for j in range(self.p_order):
                        x_local = cas.vertcat(
                            *[x_col[i][j][k] for k in range(self.x_dim)]
                        )
                        u_local = cas.vertcat(*u_points[i]) * col_points[
                            j
                        ] + cas.vertcat(*u_points[i + 1]) * (1 - col_points[j])
                        g.append(fg(x_local, u_local, du))

            glb += self.g_min * self.N * self.p_order
            gub += self.g_max * self.N * self.p_order

        if self.g_end is not None:
            for g_end in self.g_end:
                x_end = cas.vertcat(*[x_points[self.N][k] for k in range(self.x_dim)])
                g.append(g_end(x_end))

            glb += self.g_end_min
            gub += self.g_end_max

        if self.g_x0 is not None:
            for g_x0 in self.g_x0:
                x0 = cas.vertcat(*[x_points[0][k] for k in range(self.x_dim)])
                g.append(g_x0(x0))

            glb += self.g_x0_min
            gub += self.g_x0_max

        # get start values for solver

        for f in self.end_guess:
            f.terminal = True

        x0_ode = (np.array(self.x0_min) + np.array(self.x0_max)) / 2

        def f(t, x):
            return self.f(
                x,
                self.u_guess(t),
                self.u_discrete_guess,
                self.du_guess(t),
                x0_ode,
            )

        sol = scipy.integrate.solve_ivp(
            fun=f,
            t_span=(0, self.t_max),
            y0=x0_ode,
            dense_output=True,
            events=self.end_guess,
            **self.ode_solver_options,
        )
        t_end_guess = sol.t[-1]
        t_guess = t_values * t_end_guess
        t_interim = [
            t + t_end_guess / self.N * col
            for col in np.array(B).reshape(-1)
            for t in t_guess[:-1]
        ]

        points = sol.sol(t_guess).reshape(-1, order="F")
        interim = sol.sol(t_interim).reshape(-1, order="F")

        # print(points)

        u_values = np.array([self.u_guess(t) for t in t_guess]).reshape(-1)

        x0_guess = np.hstack(
            (points, interim, u_values, self.u_discrete_guess, t_end_guess)
        )

        # create objects needed for solver

        x_solver = cas.vertcat(*x_all)
        g_solver = cas.vertcat(*g)

        lbx_solver = cas.vertcat(*lbx)
        ubx_solver = cas.vertcat(*ubx)
        lbg_solver = cas.vertcat(*glb)
        ubg_solver = cas.vertcat(*gub)

        x0_solver = cas.vertcat(*x0_guess)

        self.parameter = {"f": J, "x": x_solver, "g": g_solver}
        self.bounds = {
            "lbx": lbx_solver,
            "ubx": ubx_solver,
            "lbg": lbg_solver,
            "ubg": ubg_solver,
            "x0": x0_solver,
        }

        solver = cas.nlpsol("solver", "ipopt", self.parameter, self.solver_options)

        sol = solver(**self.bounds)

        # extract dataframe from solution object

        x_p = list()
        x_c = list()
        u_p = list()

        t_end_value = float((sol["x"])[-1])
        u_discrete = [
            float(sol["x"][i]) for i in range(-1 - len(self.u_discrete_min), -1)
        ]
        t = t_values * t_end_value

        t_col = (
            np.array([t + col / self.N for col in col_points for t in t_values[:-1]])
            * t_end_value
        )

        for k in range(self.x_dim):
            x_points_eval = cas.vertcat(*[x_points[i][k] for i in range(self.N + 1)])
            x_col_eval = cas.vertcat(
                *[x_col[i][j][k] for j in range(self.p_order) for i in range(self.N)]
            )
            eval = cas.Function("eval", [x_solver], [x_points_eval, x_col_eval])
            x_point_values, x_col_values = eval(sol["x"])
            x_p.append(np.array(x_point_values).reshape(-1))
            x_c.append(np.array(x_col_values).reshape(-1))

        for k in range(self.u_dim):
            u_eval = cas.vertcat(*[u_points[i][k] for i in range(self.N + 1)])
            eval = cas.Function("eval", [x_solver], [u_eval])
            u_p.append(np.array(eval(sol["x"])).reshape(-1))

        x_p = np.array(x_p).T
        x_c = np.array(x_c).T
        u_p = np.array(u_p).T

        # set column names
        if self.x_names is not None:
            x_names = self.x_names
        else:
            x_names = [f"x_{j}" for j in range(self.x_dim)]

        if self.u_names is not None:
            u_names = self.u_names
        else:
            u_names = [f"u_{k}" for k in range(self.u_dim)]

        self.df_time_points = pd.DataFrame(
            np.hstack((x_p, u_p)), columns=x_names + u_names, index=t
        )

        self.df_intermediates = pd.DataFrame(
            x_c,
            columns=x_names,
            index=t_col,
        )

        self.df = pd.concat((self.df_time_points, self.df_intermediates))
        self.df.sort_index(inplace=True)
        self.u_discrete_value = u_discrete

    def plot_variables(self, labels=None, figsize=(15, 7.5)):
        no_plots = len(self.df.columns)
        n_col = 4
        if no_plots % n_col == 0:
            n_row = no_plots // n_col
        else:
            n_row = no_plots // n_col + 1

        fig, axs = plt.subplots(n_row, n_col, figsize=figsize)

        if labels is None:
            labels = self.df.columns

        for ax, col, label in zip(axs.reshape(-1), self.df, labels):
            ax.plot(self.df[col][~np.isnan(self.df[col])])
            ax.set_ylabel(label)
            ax.set_xlabel("t [h]")

        fig.tight_layout()

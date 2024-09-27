# fits the model and returns differential equations for this model
#
# usage:
# fit_model(df, Y_XG, Y_PG)

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.integrate
import logging

# physical constants
R = 8.314  # gas constant
kB = 1.38e-23  # boltzmann constant
h = 6.63e-34  # planck constant


class FBase:
    """base feed requirement dependend on mu"""

    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def f_base(self, mu):
        return self.slope * mu + self.intercept


def f_quad_error(x, y):
    return np.sum((x - y) ** 2)


def f_abs_error(x, y):
    return np.sum(np.abs(x - y))


def f_quad_error_rel(x, y):
    return np.sum(((x - y) / (x + y) / 2) ** 2)


class Fit:
    def __init__(
        self,
        f,
        xdata,
        ydata,
        p0,
        f_error,
        bounds=None,
        maxiter=10000,
    ) -> None:
        self.f = f
        self.xdata = xdata
        self.ydata = ydata
        self.f_error = f_error
        self.p0 = p0
        if bounds is not None:
            lower = np.array([x for (x, y) in bounds])
            upper = np.array([y for (x, y) in bounds])
            self.bounds = [(x, y) for (x, y) in zip(lower, upper)]
        else:
            self.bounds = [(-np.inf, np.inf)] * len(self.p0)
        self.maxiter = maxiter

    def f_minimizer(self, param):
        return self.f_error(
            self.f(self.xdata, *param),
            self.ydata,
        )

    def minimize(self, tol=1e-12):
        res = scipy.optimize.minimize(
            fun=self.f_minimizer, x0=self.p0, bounds=self.bounds
        )
        return res

    def basinhopping(self, tol=1e-10, niter=100):
        res = scipy.optimize.basinhopping(
            func=self.f_minimizer,
            x0=self.p0,
            minimizer_kwargs={"bounds": self.bounds, "tol": tol},
            niter=niter,
        )
        return res

    def differential_evolution(
        self,
        de_options={
            "tol": 1e-5,
        },
    ):
        res = scipy.optimize.differential_evolution(
            func=self.f_minimizer,
            x0=self.p0,
            bounds=self.bounds,
            maxiter=self.maxiter,
            # recombination=0.6,
            **de_options,
        )
        return res

    def brute(self, Ns=5):
        opt, fval, grid, Jout = scipy.optimize.brute(
            func=self.f_minimizer,
            ranges=self.bounds,
            Ns=Ns,
            full_output=True,
        )
        return fval


def f_i(x, ki_x):
    if ki_x == 0:
        return 0
    else:
        return 1 / (1 + x / ki_x)


def f_i_n_exp(x, ki_x):
    return 1 / (1 + 2**x / ki_x)


def f_MM(x, km):
    return x / (km + x)


def f_lin(x, k):
    return 1 + x * k


def f_lin_n_exp(x, k):
    return 1 + 2**x * k


def f_T(T, dG_cat, dH_eq, T_eq):
    return (
        kB
        * T
        * np.exp(-dG_cat / (R * T))
        / (h * (1 + np.exp(dH_eq * (1 / T_eq - 1 / T) / R)))
        * 3600
    )


def calc_var(dependence, df, c_bounds=(0, 1), target="g"):
    fs = list()
    bounds = [c_bounds]
    if "T" in [x[0] for x in dependence]:
        bounds = [(0, 1e-5)]
    vars = list()

    for var, f, bound in dependence:
        fs.append(f)
        vars.append(var)
        bounds += bound

    def f_all(x, *params):
        result = params[0]
        i = 1
        for j, (f, d) in enumerate(zip(fs, dependence)):
            no_params = len(d[2])
            result *= f(x[:, j], *params[i : i + no_params])
            i += no_params
        return result

    xdata = df[vars].values
    ydata = df[target].values

    bounds = np.array(bounds)
    p0 = np.mean(bounds, axis=1)

    min = Fit(f_all, xdata, ydata, p0, f_quad_error, bounds=bounds)
    opt = min.differential_evolution(
        de_options={
            "tol": 1e-5,
        }
    )
    if opt.success == False:
        logging.info(f"differntial evolution did not converge\n{opt}")
    var = opt.fun
    param = opt.x
    errors = f_all(xdata, *param) - ydata
    return opt, var, errors


def calc_var_table(dependence, df, target, c_bound=[0, 1]):
    opt_all, var_all, errors_all = calc_var(dependence, df, c_bound, target)
    params = opt_all.x
    params_grouped = [[params[0]]]
    i = 1
    for var, f, bounds in dependence:
        no_params = len(bounds)
        params_grouped.append(params[i : i + no_params])
        i += no_params

    results = [["const", [params[0]], 0, var_all, 0]]
    for i in range(len(dependence)):
        dep = [dependence[j] for j in range(len(dependence)) if j != i]
        opt, var, errors = calc_var(dep, df, c_bound, target)
        f = np.var(errors, ddof=1) / np.var(errors_all, ddof=1)
        nun = errors.size - 1
        dun = errors_all.size - 1
        p_value = 1 - scipy.stats.f.cdf(f, nun, dun)
        results.append(
            [dependence[i][0], params_grouped[i + 1], p_value, var, var - var_all]
        )
    df_var = pd.DataFrame(
        results,
        columns=[
            "variable",
            "parameters",
            "p_value",
            "variance",
            "explained_variance",
        ],
    )
    return results, df_var


class OdeG:
    def __init__(self, dep, res) -> None:
        self.dep = dep
        self.params = [x[1] for x in res]

    def f_g(self, G, n, P, X, T):
        param_dict = {"X": X, "P_X": P / X, "n": n, "T": T, "G": G, "Gi": G}
        result = self.params[0][0]
        for i, d in enumerate(self.dep):
            f = d[1]
            xi = param_dict[d[0]]
            result *= f(xi, *self.params[i + 1])
        return result


class OdeGgmax:
    def __init__(self) -> None:
        self.g_max = 0.5
        self.km = 1e-3

    def f_g(self, G, n, P, X, T):
        return self.g_max * G / (self.km + G)


class OdeGm:
    def __init__(self, dep, res) -> None:
        self.dep = dep
        self.params = [x[1] for x in res]

    def f_gm(self, g, G, n, P, X, T):
        param_dict = {"X": X, "P_X": P / X, "n": n, "T": T, "g": g, "G": G}
        result = self.params[0][0]
        for i, d in enumerate(self.dep):
            f = d[1]
            xi = param_dict[d[0]]
            result *= f(xi, *self.params[i + 1])
        return result


class OdeGp:
    def __init__(self, dep, res) -> None:
        self.dep = dep
        self.params = [x[1] for x in res]

    def f_gP(self, g, gm, G, n, P, X, T):
        param_dict = {"X": X, "P_X": P / X, "n": n, "T": T, "g_gm": g - gm, "G": G}
        result = self.params[0][0]
        for i, d in enumerate(self.dep):
            f = d[1]
            xi = param_dict[d[0]]
            result *= f(xi, *self.params[i + 1])
        return result


class Ode:
    def __init__(
        self, f_g: OdeG, f_gm: OdeGm, f_gP: OdeGp, Y_XG: float, Y_PG: float
    ) -> None:
        self.f_g = f_g.f_g
        self.f_gm = f_gm.f_gm
        self.f_gP = f_gP.f_gP
        self.Y_XG = Y_XG
        self.Y_PG = Y_PG

    def f(self, x, u, x0):
        X = x[0]
        P = x[1]
        G = x[2]
        V = x[3]
        f = u[0]
        c_f = u[1]
        T = u[2]
        X0 = x0[0]
        V0 = x0[3]
        G = np.fmax(G, 0)
        P = np.fmax(P, 0)
        n = np.log(X * V / (X0 * V0)) / np.log(2)
        g = self.f_g(G, n, P, X, T)
        gm = self.f_gm(g, G, n, P, X, T)
        gm = np.fmin(gm, g)
        gP = self.f_gP(g, gm, G, n, P, X, T)
        gP = np.fmin(gP, g - gm)
        g_mu = g - gm - gP
        mu = (g_mu * self.Y_XG + gP * self.Y_PG) * (1 - P / X)
        qP = gP * self.Y_PG * (1 - P / X)
        return [
            mu * X - f / V * X,
            qP * X - f / V * P,
            -g * (X - P) + f / V * (c_f - G),
            f,
        ]


def get_params(df: pd.DataFrame, dependence: list, p=0.2, target="g"):
    # possible functions

    def get_ode(dep, res, target):
        d = dep.copy()
        r = res.copy()
        if target == "g":
            if "G" not in [var_name for var_name, _, _ in dep]:
                d.append(["G", f_MM, [(1e-15, 1e2)]])
                r.append(["G", [1e-3], 0, 0, 0])
            ode = OdeG(d, r)
        elif target == "gm":
            ode = OdeGm(d, r)
        else:  # target == "gP"
            ode = OdeGp(d, r)
        return ode

    res, df_var = calc_var_table(dependence, df, target)

    df_var_all = df_var.copy()
    dep = dependence.copy()
    df_var_steps = [df_var_all]
    odes = list()
    odes.append(get_ode(dep, res, target))

    while df_var.p_value.max() > p and len(df_var.index) > 1:
        i = df_var.p_value.argmax()
        dep = [x for j, x in enumerate(dep) if j != i - 1]
        res, df_var = calc_var_table(dep, df, target)
        df_var_steps.append(df_var)
        odes.append(get_ode(dep, res, target))

    return res, dep, df_var, df_var_all, df_var_steps, odes


def fit_model(
    df: pd.DataFrame, Y_XG: float, Y_PG: float, G_not_measureable=True, alpha=0.2
) -> tuple[Ode, dict[str, pd.DataFrame]]:
    # g
    dfs_var = dict()
    if df.G.max() < 1e-1:
        ode_g = OdeGgmax()
    else:
        if G_not_measureable:
            df_g = df[df.G > 1e-1].copy()
        else:
            df_g = df.copy()

        # create new variable for G inhibition in contrast to G Michaelis-Menten

        df_g["Gi"] = df_g.G

        dependence = [
            ["G", f_MM, [(1e-15, 1e2)]],
            ["X", f_i, [(0, 1e4)]],
            ["P_X", f_i, [(0, 1e4)]],
            ["n", f_i, [(0, 1e4)]],
            # ["n", f_i_n_exp, [(0, 1e4)]],
            ["Gi", f_i, [(1e-5, 1e4)]],
            ["T", f_T, [(4e4, 1e5), (1e5, 1e7), (300, 315)]],
        ]

        (
            res,
            dep,
            dfs_var["g"],
            dfs_var["g_full"],
            dfs_var["g_steps"],
            dfs_var["g_odes"],
        ) = get_params(df_g, dependence, target="g", p=alpha)
        # set Km to 1e-3 if no Km is found
        if "G" not in [var_name for var_name, _, _ in dep]:
            dep.append(["G", f_MM, [(1e-15, 1e2)]])
            res.append(["G", [1e-3], 0, 0, 0])
        ode_g = OdeG(dep, res)
    # f_g = get_ode_g(dep, res)

    dependence = [
        ["g", f_lin, [(0, 1e6)]],
        ["X", f_lin, [(0, 1e6)]],
        ["P_X", f_lin, [(0, 1e6)]],
        ["n", f_lin, [(0, 1e6)]],
        ["G", f_lin, [(0, 1e6)]],
        ["T", f_T, [(4e4, 1e5), (1e5, 1e7), (300, 315)]],
    ]

    (
        res,
        dep,
        dfs_var["gm"],
        dfs_var["gm_full"],
        dfs_var["gm_steps"],
        dfs_var["gm_odes"],
    ) = get_params(
        df,
        dependence,
        target="gm",
        p=alpha,
    )
    ode_gm = OdeGm(dep, res)
    # f_gm = get_ode_gm(dep, res)

    dependence = [
        ["g_gm", f_MM, [(0, 1e2)]],
        ["X", f_i, [(0, 1e4)]],
        ["P_X", f_i, [(0, 1e4)]],
        ["n", f_i, [(0, 1e4)]],
        ["G", f_i, [(0, 1e4)]],
        ["T", f_T, [(1e4, 1e5), (1e5, 1e7), (300, 315)]],
    ]

    (
        res,
        dep,
        dfs_var["gP"],
        dfs_var["gP_full"],
        dfs_var["gP_steps"],
        dfs_var["gP_odes"],
    ) = get_params(
        df,
        dependence,
        target="gP",
        p=alpha,
    )
    ode_gP = OdeGp(dep, res)
    # f_gP = get_ode_gP(dep, res)

    ode = Ode(ode_g, ode_gm, ode_gP, Y_XG, Y_PG)
    # f_ode = get_f_ode(f_g, f_gm, f_gP, Y_XG, Y_PG)

    return ode, dfs_var


def cross_validate(
    df: pd.DataFrame,
    data_df: pd.DataFrame,
    test_process: str,
    f_spline: dict,
    ode: Ode,
    Y_XG: float,
    Y_PG: float,
    alpha: float,
    t_plot=np.linspace(0, 12.5, 1000),
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, Ode]:
    df_CV = df[df.process != test_process]
    df_test = data_df[data_df.process == test_process]
    ode_CV, dfs_var = fit_model(df_CV, Y_XG, Y_PG, alpha=alpha)

    x0 = [df_test.X.iloc[0], df_test.P.iloc[0], df_test.G.iloc[0], df_test.V.iloc[0]]

    def u(t):
        return [
            f_spline[test_process].f(t) / f_spline[test_process].sampling_factor(t),
            f_spline[test_process].c_feed(t),
            f_spline[test_process].T(t),
        ]

    def f(t, x):
        return ode.f(x, u(t), x0)

    def f_CV(t, x):
        return ode_CV.f(x, u(t), x0)

    sol = scipy.integrate.solve_ivp(
        f,
        t_span=[0, df_test.t.values[-1]],
        y0=x0,
        t_eval=df_test.t.values,
        method="BDF",
    )

    sol_CV = scipy.integrate.solve_ivp(
        f_CV,
        t_span=[0, df_test.t.values[-1]],
        y0=x0,
        t_eval=df_test.t.values,
        method="BDF",
    )

    df_CV = data_df[data_df.process == test_process].copy()
    df_CV["X_est"] = sol.y[0]
    df_CV["P_est"] = sol.y[1]
    df_CV["G_est"] = sol.y[2]
    df_CV["V_est"] = sol.y[3]
    df_CV["X_est_CV"] = sol_CV.y[0]
    df_CV["P_est_CV"] = sol_CV.y[1]
    df_CV["G_est_CV"] = sol_CV.y[2]
    df_CV["V_est_CV"] = sol_CV.y[3]

    sol = scipy.integrate.solve_ivp(
        f,
        t_span=[0, t_plot[-1]],
        y0=x0,
        t_eval=t_plot,
        method="BDF",
    )

    sol_CV = scipy.integrate.solve_ivp(
        f_CV,
        t_span=[0, t_plot[-1]],
        y0=x0,
        t_eval=t_plot,
        method="BDF",
    )

    # if (len(t_plot) != len(sol.y[0])) or (len(t_plot) != len(sol_CV.y[0])):
    #     print(sol, sol_CV)
    #     return (None, None, None, None)

    df_CV_plot = pd.DataFrame(
        {
            "t": t_plot,
            "X_est": sol.y[0],
            "P_est": sol.y[1],
            "G_est": sol.y[2],
            "X_est_CV": sol_CV.y[0],
            "P_est_CV": sol_CV.y[1],
            "G_est_CV": sol_CV.y[2],
            "process": test_process,
        }
    )
    df_CV_plot["T"] = [
        f_spline[process].T(t) for process, t in zip(df_CV_plot.process, df_CV_plot.t)
    ]

    return dfs_var, df_CV, df_CV_plot, ode_CV

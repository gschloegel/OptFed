from os import replace
import numpy as np
import pandas as pd
import scipy.integrate
import logging
import fit_model
import typing

# physical constants
R = 8.314  # gas constant
kB = 1.38e-23  # boltzmann constant
h = 6.63e-34  # planck constant
random = np.random.default_rng(0)


def create_params(
    no_params: int,  # [0, 15]
    G_max: float,
    n_max: float,
    PX_max: float,
    X_max: float,
    K_m_range=[0, 1],
    # it meight be necessery to change the differential equation constraints if values are changed
    T_eq_range=[304.15, 308.15],
    G_cat_range=[4e4, 1e5],
    H_eq_range=[7e4, 1e6],
    # maximal values at 304.15 K
    g_max_range=[0.3, 1],
    gm0_range=[0, 0.05],
    gP_max_range=[0, 0.1],
) -> dict[str, float]:
    """
    creates a random set of parameter for the model
    K_m, k_g and kappa_m are always part of the model
    either g_max, gm0, gP_max or the temperature effects are part ofthe model
    no_param specifies how many additional factors are in the model
    the inhibition parameters are selected, that the median of the distribution
    halfs g, gP and doubles gm at the highest ovserved values in the data
    """
    T0 = 304.15
    params = dict()
    # the median of the distribution halfs the rates at maximal observed concentrations
    # g
    params["g_max"] = random.uniform(g_max_range[0], g_max_range[1])
    params["K_m"] = random.uniform(K_m_range[0], K_m_range[1])
    params["K_G"] = random.exponential(1 / (np.log(2) / G_max))
    params["K_n"] = random.exponential(1 / (np.log(2) / n_max))
    params["K_P"] = random.exponential(1 / (np.log(2) / PX_max))
    params["K_X"] = random.exponential(1 / (np.log(2) / X_max))
    params["G_cat_g"] = random.uniform(G_cat_range[0], G_cat_range[1])
    params["H_eq_g"] = random.uniform(H_eq_range[0], H_eq_range[1])
    params["T_eq_g"] = random.uniform(T_eq_range[0], T_eq_range[1])
    # gm
    params["gm0"] = random.uniform(gm0_range[0], gm0_range[1])
    # adaption "/ gm0" is necessary as gm0 * (1 + k_g * g) = gm0 + k_g * g * gm0
    # the k_g * gm0 * g term should be less than half the total uptake
    # k_g * gm0 < 1/2
    params["k_g"] = random.uniform(0, 1 / params["gm0"] / 2)
    # params["k_g"] = random.exponential(np.log(2) * g_max_range[1])
    params["k_G"] = random.exponential(1 / (np.log(2) * G_max))
    params["k_n"] = random.exponential(1 / (np.log(2) * n_max))
    params["k_P"] = random.exponential(1 / (np.log(2) * PX_max))
    params["k_X"] = random.exponential(1 / (np.log(2) * X_max))
    params["G_cat_gm"] = random.uniform(G_cat_range[0], G_cat_range[1])
    params["H_eq_gm"] = random.uniform(H_eq_range[0], H_eq_range[1])
    params["T_eq_gm"] = random.uniform(T_eq_range[0], T_eq_range[1])
    # gP
    params["gP_max"] = random.uniform(gP_max_range[0], gP_max_range[1])
    params["kappa_m"] = random.exponential(g_max_range[1])
    params["kappa_G"] = random.exponential(1 / (np.log(2) / G_max))
    params["kappa_n"] = random.exponential(1 / (np.log(2) / n_max))
    params["kappa_P"] = random.exponential(1 / (np.log(2) / PX_max))
    params["kappa_X"] = random.exponential(1 / (np.log(2) / X_max))
    params["G_cat_gP"] = random.uniform(G_cat_range[0], G_cat_range[1])
    params["H_eq_gP"] = random.uniform(H_eq_range[0], H_eq_range[1])
    params["T_eq_gP"] = random.uniform(T_eq_range[0], T_eq_range[1])

    # E0 is selected that g_max(304.15) = g_max
    params["E0_g"] = params["g_max"] / (
        kB
        * T0
        / h
        * np.exp(-params["G_cat_g"] / (R * T0))
        / (1 + np.exp(params["H_eq_g"] / R * (1 / params["T_eq_g"] - 1 / T0)))
        * 3600
    )
    params["E0_gm"] = params["gm0"] / (
        kB
        * T0
        / h
        * np.exp(-params["G_cat_gm"] / (R * T0))
        / (1 + np.exp(params["H_eq_gm"] / R * (1 / params["T_eq_gm"] - 1 / T0)))
        * 3600
    )
    params["E0_gP"] = params["gP_max"] / (
        kB
        * T0
        / h
        * np.exp(-params["G_cat_gP"] / (R * T0))
        / (1 + np.exp(params["H_eq_gP"] / R * (1 / params["T_eq_gP"] - 1 / T0)))
        * 3600
    )

    # setting values to 0 to keep just no_params values
    # g_max, gm_0, gP_max, k_g, and kappa_m are always in the value and do not count for no_params
    param_list = [
        "K_G",
        "K_n",
        "K_P",
        "K_X",
        "E0_g",
        "k_G",
        "k_n",
        "k_P",
        "k_X",
        "E0_gm",
        "kappa_G",
        "kappa_n",
        "kappa_P",
        "kappa_X",
        "E0_gP",
    ]
    to_delete = random.choice(
        a=np.array(param_list), size=len(param_list) - no_params, replace=False
    )
    for x in to_delete:
        if x in ["k_G", "k_n", "k_P", "k_X", "E0_g", "E0_gm", "E0_gP"]:
            params[x] = 0
        else:
            params[x] = np.inf

    # make sure either g_max is a parameter of the temperature effect
    if params["E0_g"] != 0:
        params["g_max"] = 0
    else:
        params["G_cat_g"] = 0
        params["H_eq_g"] = 0
        params["T_eq_g"] = 0

    if params["E0_gm"] != 0:
        params["gm0"] = 0
    else:
        params["G_cat_gm"] = 0
        params["H_eq_gm"] = 0
        params["T_eq_gm"] = 0

    if params["E0_gP"] != 0:
        params["gP_max"] = 0
    else:
        params["G_cat_gP"] = 0
        params["H_eq_gP"] = 0
        params["T_eq_gP"] = 0

    return params


class Ode:
    """returns model ODE from parameter dict"""

    def __init__(
        self,
        params: dict,
        X0,
        Y_XG: float,
        Y_PG: float,
        c_feed: float,
        u_as_uptake=False,
        k_f_base=0.00138240065229037,
        d_f_base=6.027340556575997e-07,
    ) -> None:
        self.params = params
        self.X0 = X0
        self.Y_XG = Y_XG
        self.Y_PG = Y_PG
        self.c_feed = c_feed
        self.u_as_uptake = u_as_uptake
        self.k_f_base = k_f_base
        self.d_f_base = d_f_base

        def eliminate_nan_from_params(params):
            for key, value in params.items():
                if np.isnan(value):
                    if key in [
                        "K_m",
                        "k_g",
                        "k_G",
                        "k_n",
                        "k_P",
                        "k_X",
                        "E0_g",
                        "E0_gm",
                        "E0_gP",
                        "g_max",
                        "gm0",
                        "gP_max",
                        "kappa_m",
                    ]:
                        params[key] = 0
                    else:
                        params[key] = np.inf
            if params["K_m"] == 0:
                params["K_m"] = 1e-3
            return params

        self.params = eliminate_nan_from_params(self.params)

    def f(self, x, u):
        X = x[0]
        P = x[1]
        G = x[2]
        V = x[3]

        if self.u_as_uptake:
            g = u[0]
            T = u[1]
            f = g * X * V / self.c_feed
        else:
            f = u[0]
            T = u[1]

        G = np.fmax(0, G)
        X = np.fmax(0, X)
        P = np.fmax(0, P)
        n = np.log(X * V / self.X0) / np.log(2)
        n = np.fmax(0, n)
        if self.params["g_max"] == 0:
            g_max = (
                self.params["E0_g"]
                * kB
                * T
                / h
                * np.exp(-self.params["G_cat_g"] / (R * T))
                / (
                    1
                    + np.exp(
                        self.params["H_eq_g"] / R * (1 / self.params["T_eq_g"] - 1 / T)
                    )
                )
                * 3600
            )
        else:
            g_max = self.params["g_max"]

        if self.params["gm0"] == 0:
            gm0 = (
                self.params["E0_gm"]
                * kB
                * T
                / h
                * np.exp(-self.params["G_cat_gm"] / (R * T))
                / (
                    1
                    + np.exp(
                        self.params["H_eq_gm"]
                        / R
                        * (1 / self.params["T_eq_gm"] - 1 / T)
                    )
                )
                * 3600
            )
        else:
            gm0 = self.params["gm0"]

        if self.params["gP_max"] == 0:
            gP_max = (
                self.params["E0_gP"]
                * kB
                * T
                / h
                * np.exp(-self.params["G_cat_gP"] / (R * T))
                / (
                    1
                    + np.exp(
                        self.params["H_eq_gP"]
                        / R
                        * (1 / self.params["T_eq_gP"] - 1 / T)
                    )
                )
                * 3600
            )
        else:
            gP_max = self.params["gP_max"]

        # assert gP_max > 0, f"gP_max = {gP_max}"
        # assert X > 0, f"X = {X}"

        g = (
            g_max
            * G
            / (self.params["K_m"] + G)
            / (1 + G / self.params["K_G"])
            / (1 + n / self.params["K_n"])
            / (1 + P / X / self.params["K_P"])
            / (1 + X / self.params["K_X"])
        )
        gm = (
            gm0
            * (1 + g * self.params["k_g"])
            * (1 + G * self.params["k_G"])
            * (1 + n * self.params["k_n"])
            * (1 + P / X * self.params["k_P"])
            * (1 + X * self.params["k_X"])
        )
        gm = np.fmin(gm, g)
        if g - gm <= 0:
            gP = 0
        else:
            gP = (
                gP_max
                * (g - gm)
                / (self.params["kappa_m"] + g - gm)
                / (1 + G / self.params["kappa_G"])
                / (1 + n / self.params["kappa_n"])
                / (1 + P / X / self.params["kappa_P"])
                / (1 + X / self.params["kappa_X"])
            )
        # if np.isnan(gP):
        #     print(
        #         gP,
        #         g,
        #         gm,
        #         G,
        #         n,
        #         P,
        #         X,
        #         gP_max,
        #         self.params["kappa_m"],
        #         self.params["kappa_G"],
        #         self.params["kappa_n"],
        #         self.params["kappa_P"],
        #         self.params["kappa_X"],
        #     )
        gP = np.fmin(gP, g - gm)
        g_mu = g - gm - gP
        g_mu = np.fmax(0, g_mu)
        mu = (g_mu * self.Y_XG + gP * self.Y_PG) * (1 - P / X)
        qP = gP * self.Y_PG * (1 - P / X)
        # f_real includes the predicted base feed
        f_real = f + X * V * (self.k_f_base * mu + self.d_f_base)
        c_feed_real = self.c_feed * f / f_real
        return [
            mu * X - f_real / V * X,
            qP * X - f_real / V * P,
            -g * (X - P) + f_real / V * (c_feed_real - G),
            f,
        ]


def sim_processes(
    params: dict[str, float],
    X0: typing.Union[float, np.floating],
    Y_XG: float,
    Y_PG: float,
    c_feed: float,
    k_f_base=0.00138240065229037,
    d_f_base=6.027340556575997e-07,
    us=[
        [0.15, 304.15],
        [0.15, 304.15],
        [0.15, 304.15],
        [0.15, 304.15],
        [0.05, 300.15],
        [0.05, 308.15],
        [0.25, 300.15],
        [0.25, 308.15],
        [0.009, 304.15],
        [0.291, 304.15],
        [0.15, 298.5],
        [0.15, 309.8],
    ],
    ode_solver="BDF",
    ode_options=dict(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """simulates 12 processes following the RSM scheme of the real data"""
    # when 2 values in us are given they are for mu_feed and T (X0 is constants)
    # for 3 values they are mu_feed, T, and X0 (X0 in the function definition is ignored)
    if len(us[0]) == 2:
        us = [[u[0], u[1], X0] for u in us]

    no_processes = len(us)
    V0 = 1

    res = list()
    res_Vf = list()

    # print(list(enumerate(us))[15])

    for i, (mu, T, X0) in enumerate(us):
        ode = Ode(params, X0, Y_XG, Y_PG, c_feed, k_f_base=k_f_base, d_f_base=d_f_base)
        f_ode = ode.f

        def f(t, x):
            # for feed assume mu ~ g * Y_XG
            f0 = (X0 * mu) / (c_feed * Y_XG)
            feed = f0 * np.exp(mu * t)
            return f_ode(x, [feed, T])

        sol = scipy.integrate.solve_ivp(
            fun=f,
            t_span=[0, 12],
            y0=[X0, 0, 0, V0],
            method=ode_solver,
            t_eval=[0, 2, 4, 6, 8, 10, 12],
            dense_output=True,
            **ode_options,
        )
        if sol.status != 0:
            return None, None
        res.append(sol.y)
        ts = np.linspace(0, 12, 12 * 100 + 1)
        f0 = (X0 * mu) / (c_feed * Y_XG)
        V = sol.sol(ts)[3]
        f_cum = V - V0
        f_cum_substrate = f0 / mu * (np.exp(mu * ts) - 1)
        f_cum_base = f_cum - f_cum_substrate
        res_Vf.append(
            pd.DataFrame(
                {
                    "t": ts,
                    "V": V,
                    "f_cum": f_cum_substrate,
                    "f_base_cum": f_cum_base,
                    "process": f"p{i}",
                }
            )
        )

    res = np.hstack(res)
    df = pd.DataFrame(
        {
            "X": res[0],
            "P": res[1],
            "G": res[2],
            "V": res[3],
            "T": [T for (_, T, _) in us for i in range(7)],
            "t": [0, 2, 4, 6, 8, 10, 12] * no_processes,
            "process": [f"p{i}" for i in range(no_processes) for j in range(7)],
            "mu_feed": [mu for (mu, _, _) in us for i in range(7)],
            "X0": [X0 for (_, _, X0) in us for i in range(7)],
        }
    )
    df_Vf = pd.concat(res_Vf, ignore_index=True)

    return df, df_Vf


def add_errors(
    df,
    sigma_G_abs=0,
    sigma_G_rel=0.01,
    sigma_X_abs=0,
    sigma_X_rel=0.01,
    sigma_P_abs=0,
    sigma_P_rel=0.01,
):
    """adds errors to the simulated data
    relative and absolute normal distributed errors are possible"""
    df = df.copy()
    no_data = len(df.index)
    G_abs_error = random.normal(0, sigma_G_abs, no_data)
    G_rel_error = random.normal(0, sigma_G_rel, no_data)
    X_abs_error = random.normal(0, sigma_X_abs, no_data)
    X_rel_error = random.normal(0, sigma_X_rel, no_data)
    P_abs_error = random.normal(0, sigma_P_abs, no_data)
    P_rel_error = random.normal(0, sigma_P_rel, no_data)
    df["G"] = np.maximum(0, df.G + G_abs_error + G_rel_error * df.G)
    df["X"] = np.maximum(0, df.X + X_abs_error + X_rel_error * df.X)
    df["P"] = np.maximum(0, df.P + P_abs_error + P_rel_error * df.P)
    return df


# calculate model
# for the real data the ODE for uptake was not calculated with the data
# as the substrate concentrations were below the limit of quantification.
# therefore we need to add this function here.


def get_ode_g(dep, res):
    params = [x[1] for x in res]

    def f_ode(x, u, x0):
        X, P, G, V = x
        f, c_f, T = u
        X0, P0, G0, V0 = x0
        P_X = P / X
        n = np.log(X * V / (X0 * V0)) / np.log(2)
        param_dict = {"X": X, "P_X": P_X, "n": n, "T": T, "G": G}
        result = params[0][0]
        for i, d in enumerate(dep):
            f = d[1]
            xi = param_dict[d[0]]
            result *= f(xi, *params[i + 1])
        return result

    return f_ode


def compare_estimates_with_real_data(
    params: dict[str, float], df_vars: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, list, list, list]:
    df = pd.DataFrame(pd.Series(params), columns=["real_values"])
    df["estimated_values"] = 0

    dict_g = {
        "const": "g_max",
        "G": "K_m",
        "X": "K_X",
        "P_X": "K_P",
        "n": "k_n",
        "Gi": "K_G",
    }
    if "g" in df_vars:
        for i, x in df_vars["g"].iterrows():
            if x.variable == "T":
                df.loc["G_cat_g", "estimated_values"] = x.parameters[0]
                df.loc["H_eq_g", "estimated_values"] = x.parameters[1]
                df.loc["T_eq_g", "estimated_values"] = x.parameters[2]
                df.loc["E0_g", "estimated_values"] = df.estimated_values.loc["g_max"]
                df.loc["g_max", "estimated_values"] = 0
            else:
                df.loc[dict_g[x.variable], "estimated_values"] = x.parameters[0]

    dict_gm = {
        "const": "gm0",
        "g": "k_g",
        "X": "k_X",
        "P_X": "k_P",
        "n": "k_n",
        "G": "k_G",
    }
    for i, x in df_vars["gm"].iterrows():
        if x.variable == "T":
            df.loc["G_cat_gm", "estimated_values"] = x.parameters[0]
            df.loc["H_eq_gm", "estimated_values"] = x.parameters[1]
            df.loc["T_eq_gm", "estimated_values"] = x.parameters[2]
            df.loc["E0_gm", "estimated_values"] = df.estimated_values.loc["gm0"]
            df.loc["gm0", "estimated_values"] = 0
        else:
            df.loc[dict_gm[x.variable], "estimated_values"] = x.parameters[0]

    dict_gP = {
        "const": "gP_max",
        "g_gm": "kappa_m",
        "X": "kappa_X",
        "P_X": "kappa_P",
        "n": "kappa_n",
        "G": "kappa_G",
    }
    for i, x in df_vars["gP"].iterrows():
        if x.variable == "T":
            df.loc["G_cat_gP", "estimated_values"] = x.parameters[0]
            df.loc["H_eq_gP", "estimated_values"] = x.parameters[1]
            df.loc["T_eq_gP", "estimated_values"] = x.parameters[2]
            df.loc["E0_gP", "estimated_values"] = df.estimated_values.loc["gP_max"]
            df.loc["gP_max", "estimated_values"] = 0
        else:
            df.loc[dict_gP[x.variable], "estimated_values"] = x.parameters[0]

    df.replace([0, np.inf], np.nan, inplace=True)

    missed_vars = df[
        df.real_values.notna() & df.estimated_values.isna()
    ].index.to_list()
    additional_found_vars = df[
        df.real_values.isna() & df.estimated_values.notna()
    ].index.to_list()
    correct_vars = df[
        df.real_values.notna() & df.estimated_values.notna()
    ].index.to_list()

    return df, correct_vars, missed_vars, additional_found_vars

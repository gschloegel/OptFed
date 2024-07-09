# calculates dfs with experimatal datapoints and spline fitted rates
#
# usage:
#
# def calc_rates_df(
#     df_data, df_Vf, c_feed, Y_XG, Y_PG, k=3, s=10, df_points=np.linspace(0, 12, 25)
# ) -> df, df_dense, f
# df is the dataframe containing measurments and rates at measurement times,
# df_dense at the time points given by df_points
# f are the fitted functions given as f[process][variable](t)
# k, s are the fitting parameters for the scipy.interpolate.UnivariateSpline function


import numpy as np
import pandas as pd
import scipy.interpolate
import typing
import numpy.typing as npt
import matplotlib.pyplot as plt

Y_XG = 0.627
Y_PG = 0.652


class Rates:
    """calcultes spline functions functions from dataframes
    and experimental growth and"""

    def __init__(
        self,
        df_XPGT: pd.DataFrame,
        df_Vf: pd.DataFrame,
        c_feed: float,
        Y_XG: float,
        Y_PG: float,
        k=3,
        s=10,
        s_dense=1e1,
        df_T=None,
    ) -> None:
        self.df_XPGT = df_XPGT
        self.df_Vf = df_Vf
        self.c_feed_gly = c_feed
        self.Y_XG = Y_XG
        self.Y_PG = Y_PG
        self.k = k
        self.s = s
        self.df_T = df_T

        self.X = scipy.interpolate.UnivariateSpline(
            self.df_XPGT.t, self.df_XPGT.X, k=k, s=s, ext=3
        )
        self.P = scipy.interpolate.UnivariateSpline(
            self.df_XPGT.t, self.df_XPGT.P, k=k, s=s, ext=3
        )
        self.G = scipy.interpolate.UnivariateSpline(
            self.df_XPGT.t, self.df_XPGT.G, k=k, s=s, ext=3
        )

        if self.df_T is None:
            self.T = scipy.interpolate.UnivariateSpline(
                self.df_XPGT.t, self.df_XPGT["T"], k=k, s=s, ext=3
            )
        else:
            self.T = scipy.interpolate.UnivariateSpline(
                self.df_T.t, self.df_T["T"], k=k, s=0, ext=3
            )

        self.f_Vs = dict()
        for i in range(len(self.df_XPGT.index) - 1):
            df = self.df_Vf[
                (self.df_Vf.t > self.df_XPGT.t.iloc[i])
                & (self.df_Vf.t <= self.df_XPGT.t.iloc[i + 1])
            ]
            self.f_Vs[i] = scipy.interpolate.UnivariateSpline(
                df.t, df.V, k=k, s=s_dense, ext=3
            )

            self.f_gly_cum = scipy.interpolate.UnivariateSpline(
                df_Vf.t, df_Vf.f_cum, k=k, s=s_dense, ext=3
            )
            self.f_gly = self.f_gly_cum.derivative()
            self.f_base_cum = scipy.interpolate.UnivariateSpline(
                df_Vf.t, df_Vf.f_base_cum, k=k, s=s_dense, ext=3
            )

            self.f_base = self.f_base_cum.derivative()

            self.sampling_factors_list = pd.Series(
                np.hstack(
                    [
                        [1],
                        np.cumprod(
                            [
                                self.df_Vf[
                                    (self.df_Vf.t > t) & (self.df_Vf.t < t + 1)
                                ].V.min()
                                / V
                                for t, V in zip(
                                    self.df_XPGT.t.iloc[1:-1], self.df_XPGT.V.iloc[1:-1]
                                )
                            ]
                        ),
                    ]
                ),
                index=self.df_XPGT.t[0:-1],
            )

    def V_scalar(self, t: float) -> float:
        # on the border point always take the function before the switch
        # for t = 0 the first interval has to be taken.
        i = max(1, sum(self.df_XPGT.t < t))
        if i > len(self.df_XPGT.index) - 1:
            return self.df_Vf.V.iloc[-1]
        else:
            return self.f_Vs[i - 1](t)

    def V(self, t: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return np.vectorize(self.V_scalar)(t)

    def f(self, t):
        return self.f_gly(t) + self.f_base(t)

    def c_feed(self, t):
        return (
            np.array(self.f_gly(t))
            * self.c_feed_gly
            / np.fmax((self.f_gly(t) + self.f_base(t)), 1e-6)
        )

    def mu(self, t):
        return np.array(self.X(t, nu=1)) / self.X(t) + self.f(t) / self.V(t)

    def qP(self, t):
        return (self.P(t, nu=1) + self.f(t) / self.V(t) * self.P(t)) / self.X(t)

    def g(self, t):
        return (
            self.f(t) / self.V(t) * (self.c_feed(t) - self.G(t)) - self.G(t, nu=1)
        ) / (np.array(self.X(t)) - np.array(self.P(t)))

    def gP(self, t):
        return self.qP(t) / (1 - np.array(self.P(t)) / np.array(self.X(t))) / self.Y_PG

    def g_mu(self, t):
        return (
            (self.mu(t) - self.qP(t))
            / (1 - np.array(self.P(t)) / np.array(self.X(t)))
            / self.Y_XG
        )

    def gm(self, t):
        return self.g(t) - self.g_mu(t) - self.gP(t)

    def f_sampling_factors_scalar(self, t: float) -> float:
        i = np.fmax(1, np.sum([self.sampling_factors_list.index < t]))
        return self.sampling_factors_list.values[i - 1]

    def sampling_factor(self, t):
        return np.vectorize(self.f_sampling_factors_scalar)(t)


def calc_rates_df(
    df_data: pd.DataFrame,
    df_Vf: pd.DataFrame,
    c_feeds: dict[str, float],
    Y_XG: float,
    Y_PG: float,
    k=3,
    s=10,
    s_dense=10,
    df_points=np.linspace(0, 12, 25),
    df_T=None,
) -> tuple:
    """returns 2 dataframes and rate functions for all process"""
    dfs = list()
    dfs_dense = list()
    f = dict()  # f[process][func](t)
    process_names = df_data.process.unique()
    for process_name in process_names:
        df_XPGT = df_data[df_data.process == process_name]
        df_Vf_local = df_Vf[df_Vf.process == process_name]
        c_feed = c_feeds[process_name]
        if df_T is None:
            rates = Rates(df_XPGT, df_Vf_local, c_feed, Y_XG, Y_PG, k, s, s_dense)
        else:
            df_T_local = df_T[df_T.process == process_name]
            rates = Rates(
                df_XPGT, df_Vf_local, c_feed, Y_XG, Y_PG, k, s, s_dense, df_T=df_T_local
            )

        def create_df(t):
            df = pd.DataFrame(
                {
                    "t": t,
                    "X": rates.X(t),
                    "P": rates.P(t),
                    "G": rates.G(t),
                    "T": rates.T(t),
                    "V": rates.V(t),
                    "feed_rate": rates.f(t),
                    "c_feed": rates.c_feed(t),
                    "g": rates.g(t),
                    "g_mu": rates.g_mu(t),
                    "gP": rates.gP(t),
                    "gm": rates.gm(t),
                }
            )
            # df = np.fmax(0, df)
            df["n"] = np.log(
                df.X
                * df.V
                / (df.X.iloc[0] * df.V.iloc[0] * rates.sampling_factor(df.t))
            ) / np.log(2)
            df["P_X"] = df.P / df.X
            df["g_gm"] = df.g - df.gm
            df["process"] = process_name
            return df

        t = df_XPGT.t
        dfs.append(create_df(t))
        t = df_points
        dfs_dense.append(create_df(t))
        f[process_name] = rates

    df = pd.concat(dfs, ignore_index=True)
    df_dense = pd.concat(dfs_dense, ignore_index=True)
    return df, df_dense, f


class BaseFeedFit:
    def __init__(self, slope, intercept, r_value, p_value, std_err, df: pd.DataFrame):
        self.slope = slope
        self.intercept = intercept
        self.r_value = r_value
        self.p_value = p_value
        self.std_err = std_err
        self.df = df

    def base_feed(self, mu):
        return self.slope * mu + self.intercept

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.scatter(self.df.mu, self.df.f_base)
        x = np.linspace(-0.05, 0.3, 100)
        ax.plot(x, slope * x + intercept)
        ax.set_xlabel("growth rate")
        ax.set_ylabel("base feed per biomass [L g$^{-1}$ h$^{-1}$]")
        ax.set_xlim(-0.025, 0.225)
        ax.set_ylim(-0.00005, 0.00035)
        plt.show()

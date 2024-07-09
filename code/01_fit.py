# %%

import numpy as np
import pandas as pd
import calc_rates
import fit_model

# import rpy2
import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib.lines
import scipy.integrate
import scipy.optimize
import scipy.stats
import multiprocessing
import itertools
import logging
import pickle
import collections
import pathlib

# import numpy.random

np.random.seed(0)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

fig_path = "../figs/"
data_path = "../data/"
for path in [fig_path, data_path]:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
# plt.rcParams['savefig.dpi'] = 300

# set global constants
Y_XG = 0.627
Y_PG = 0.652
Y_MG = 73.753
R = 8.314  # gas constant
kB = 1.38e-23  # boltzmann constant
h = 6.63e-34  # planck constant
density_glycerol = 1.261
s = 10
# s_dense = 1e-2
s_dense = 10

# %%

# Import data, calculate absolute masses and correct for sampling

# process_names = [
#     "DoE1_R2",
#     "DoE1_R4",
#     "DoE2_R2",
#     "DoE2_R4",
#     "DoE3_R1",
#     "DoE1_R3",
#     "DoE2_R3",
#     "DoE3_R3",
#     "DoE3_R4",
#     "DoE3_R2",
#     "DoE2_R1",
#     "DoE1_R1",
# ]

data_df = pd.read_csv(f"{data_path}sampling_points.csv", index_col=0)
df_Vf = pd.read_csv(f"{data_path}volume_flow_rates.csv", index_col=0)
df_T = pd.read_csv(f"{data_path}temperatures.csv", index_col=0)
c_feed = pd.read_csv(f"{data_path}c_feed.csv", index_col="process").c_feed.to_dict()

process_names = data_df.process.unique()

# %%

# as we give feed and volume as masses, we have to adapt
# the feed concentration from g/L to g/kg
for k, v in c_feed.items():
    density = 1 + v / 1000 * (1 - 1 / density_glycerol)
    c_feed[k] = v / density

df_measurement_points, df_dense, f_spline = calc_rates.calc_rates_df(
    data_df, df_Vf, c_feed, Y_XG, Y_PG, k=3, s=s, df_T=df_T, s_dense=s_dense
)


logging.info("create base feed estimation")

# As long as there is no substrate accumulation in the medium
# the base feed can be estimated by the growth rate.
# For model fitting the experimental base feed is used,
# for optimization this estimation is required.


def f_base(process, t):
    return f_spline[process].f(t) * (1 - f_spline[process].c_feed(t) / c_feed[process])


df = data_df[
    data_df.process.isin(
        [
            "DoE1_R2",
            "DoE1_R4",
            "DoE2_R2",
            "DoE2_R4",
            "DoE3_R1",
            "DoE1_R3",
            "DoE2_R3",
            "DoE3_R3",
            "DoE3_R4",
        ]
    )
].copy()

df["f_base_per_biomass"] = [
    f_base(process, t) / (X * V)
    for process, t, X, V in zip(df.process, df.t, df.X, df.V)
]
df["mu"] = [f_spline[process].mu(t) for process, t in zip(df.process, df.t)]

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    df.mu, df.f_base_per_biomass
)

base_feed = calc_rates.BaseFeedFit(slope, intercept, r_value, p_value, std_err, df)

with open(f"{data_path}base_feed.pickle", "wb") as file:
    pickle.dump(base_feed, file)

# %%

# ## Spline fits
logging.info("create spline fits and data frames")

#  we see that 1 datapoint does not follow the trend. We assume a measurment error here and delete this data point.In addition w adapt the spline fitting parameters $k$ and $s$ to get smooth splines for all processes.

if 40 in data_df.index:
    data_df.drop(40, inplace=True)

df_measurement_points, df_dense, f_spline = calc_rates.calc_rates_df(
    data_df, df_Vf, c_feed, Y_XG, Y_PG, k=3, s=s, df_T=df_T, s_dense=s_dense
)

_, df_dense13, _ = calc_rates.calc_rates_df(
    data_df,
    df_Vf,
    c_feed,
    Y_XG,
    Y_PG,
    k=3,
    s=s,
    df_points=np.linspace(0, 12, 13),
    df_T=df_T,
    s_dense=s_dense,
)

_, df_dense49, _ = calc_rates.calc_rates_df(
    data_df,
    df_Vf,
    c_feed,
    Y_XG,
    Y_PG,
    k=3,
    s=s,
    df_points=np.linspace(0, 12, 49),
    df_T=df_T,
    s_dense=s_dense,
)

with open(f"{data_path}data_df.pickle", "wb") as f:
    pickle.dump(data_df, f)

with open(f"{data_path}c_feed.pickle", "wb") as f:
    pickle.dump(c_feed, f)


df_dense11 = df_dense13[(df_dense13.t > 1) & (df_dense13.t < 11)]
df_dense23 = df_dense[(df_dense.t > 1) & (df_dense.t < 11)]
df_dense47 = df_dense49[(df_dense49.t > 1) & (df_dense49.t < 11)]
df_dense5 = df_measurement_points[
    (df_measurement_points.t > 1) & (df_measurement_points.t < 11)
]

# create dict in form no_points -> df_dense
# 7 means the sampling points, 13, every hour, 25 every 1/2 hour and 49 every 1/4 hour
# using the spline fitted rates
# we will just use the measurement points later on
dfs_dense = {
    7: df_measurement_points,
    13: df_dense13,
    25: df_dense,
    49: df_dense49,
    5: df_dense5,
    11: df_dense11,
    23: df_dense23,
    47: df_dense47,
}

# pickle spline functions and dataframes
with open(f"{data_path}spline_functions.pickle", "wb") as file:
    pickle.dump(f_spline, file)

with open(f"{data_path}spline_dfs.pickle", "wb") as file:
    pickle.dump(dfs_dense, file)


# %%

logging.info("fit model")
# fit models for different alphas and evaluation points
# low values for alphas lead to simple model
# reduce the number to reduce processing times

alphas = [
    0,
    1e-4,
    2e-4,
    5e-4,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
]

# no_points = [7, 13, 25, 49, 5, 11, 23, 47]
no_points = [7]
# evaluation at measurement points (every 2 h) every h, 2 per hour and 4 per hour
# for the second half t=0 and t=12 is not used as spline fits are less relyable


def model_fit(x):
    alpha, no_points = x
    return fit_model.fit_model(dfs_dense[no_points], Y_XG, Y_PG, alpha=alpha)


x = list(itertools.product(alphas, no_points))

with multiprocessing.Pool(100) as p:
    res = p.map(model_fit, x)


dfs_vars = collections.defaultdict(dict)
odes = collections.defaultdict(dict)

x = list(itertools.product(alphas, no_points))

for (alpha, grid_points), (ode, dfs_var) in zip(x, res):
    dfs_vars[alpha][grid_points] = dfs_var
    odes[alpha][grid_points] = ode


# save functions and fit results
with open(f"{data_path}odes.pickle", "wb") as file:
    pickle.dump(odes, file)
with open(f"{data_path}dfs_fit.pickle", "wb") as file:
    pickle.dump(dfs_vars, file)


logging.info("export fit results as latex tables")

# Convert parameter tables into LaTeX format

# %%

# with open(f"{data_path}dfs_fit.pickle", "rb") as file:
#     dfs_vars = pickle.load(file)

# alphas = [
#     1e-5,
#     2e-5,
#     5e-5,
#     1e-4,
#     2e-4,
#     5e-4,
#     0.001,
#     0.002,
#     0.005,
#     0.01,
#     0.02,
#     0.05,
#     0.1,
#     0.2,
#     0.3,
#     0.4,
# ]

# no_points = [7, 13, 25, 49, 5, 11, 23, 47]


def format_table(
    df_in: pd.DataFrame, parameter_names: dict[str, str], variable_names: dict[str, str]
):
    df = df_in.copy()
    df["parameter_name"] = 0
    df = df[
        [
            "variable",
            "parameter_name",
            "parameters",
            "p_value",
            "variance",
            "explained_variance",
        ]
    ]
    for i in df.index:
        if df.loc[i, "variable"] == "T":
            parameters = df.loc[i, "parameters"]
            # df.loc[i, "variable"] = variable_names[df.loc[i, "variable"]][0]
            df.loc[i, "parameter_name"] = variable_names[df.loc[i, "variable"]][0]
            df.loc[i, "parameters"] = df.loc[i, "parameters"][0]
            df.loc[i, "parameter_name"] = parameter_names[df.loc[i, "variable"]][0]
            df.loc[i + 1] = [
                "",
                parameter_names[df.loc[i, "variable"]][1],
                parameters[1],
                "",
                "",
                "",
            ]
            df.loc[i + 2] = [
                "",
                parameter_names[df.loc[i, "variable"]][2],
                parameters[2],
                "",
                "",
                "",
            ]
            df.loc[i, "variable"] = "$T$"
            df.loc[0, "parameter_name"] = parameter_names["E0"]
        else:
            df.loc[i, "parameters"] = df.loc[i, "parameters"][0]
            # print(df.loc[i, "variable"])
            df.loc[i, "parameter_name"] = parameter_names[df.loc[i, "variable"]]

            df.loc[i, "variable"] = variable_names[df.loc[i, "variable"]]

        # df["parameters"] = pd.to_numeric(df.parameters)
    return df


parameter_names_g = {
    "const": "$\gmax$",
    "X": "$\kXg$",
    "P_X": "$\kPg$",
    "n": "$\kng$",
    "G": "$\kmg$",
    "Gi": "$\kGg$",
    "T": ["$\dGg$", "$\Heqg$", "$\Teqg$"],
    "E0": "$\Eg$",
}

parameter_names_gm = {
    "const": "$\gmmax$",
    "g": "$\kggm$",
    "X": "$\kXgm$",
    "P_X": "$\kPgm$",
    "n": "$\kngm$",
    "G": "$\kGgm$",
    "Gi": "$\kGgm$",
    "T": ["$\dGgm$", "$\Heqgm$", "$\Teqgm$"],
    "E0": "$\Egm$",
}

parameter_names_gP = {
    "const": "$\gPmax$",
    "g_gm": "$\kmgP$",
    "X": "$\kXgP$",
    "P_X": "$\kPgP$",
    "n": "$\kngP$",
    "G": "$\kGgP$",
    "Gi": "$\kGgP$",
    "T": ["$\dGgP$", "$\HeqgP$", "$\TeqgP$"],
    "E0": "$\EgP$",
}

variable_names = {
    "const": "",
    "g": "$g$",
    "g_gm": "$g - \gm$",
    "X": "$X$",
    "P_X": "$P/X$",
    "n": "$n$",
    "G": "$G$",
    "Gi": "$G$",
    "T": "$T$",
}

dfs_g = collections.defaultdict(dict)
dfs_g_full = collections.defaultdict(dict)
dfs_gm = collections.defaultdict(dict)
dfs_gm_full = collections.defaultdict(dict)
dfs_gP = collections.defaultdict(dict)
dfs_gP_full = collections.defaultdict(dict)

for alpha in alphas:
    for grid_points in no_points:
        dfs_g[alpha][grid_points] = format_table(
            dfs_vars[alpha][grid_points]["g"], parameter_names_g, variable_names
        )
        dfs_g[alpha][grid_points].loc[-1] = ["", "$\kmg$", 1e-3, np.nan, np.nan, np.nan]
        dfs_g_full[alpha][grid_points] = format_table(
            dfs_vars[alpha][grid_points]["g_full"], parameter_names_g, variable_names
        )
        dfs_gm[alpha][grid_points] = format_table(
            dfs_vars[alpha][grid_points]["gm"], parameter_names_gm, variable_names
        )
        dfs_gm_full[alpha][grid_points] = format_table(
            dfs_vars[alpha][grid_points]["gm_full"], parameter_names_gm, variable_names
        )
        dfs_gP[alpha][grid_points] = format_table(
            dfs_vars[alpha][grid_points]["gP"], parameter_names_gP, variable_names
        )
        dfs_gP_full[alpha][grid_points] = format_table(
            dfs_vars[alpha][grid_points]["gP_full"], parameter_names_gP, variable_names
        )


# %%

# Cross Validation

# As reference we calculate the cross validation for a model without any inhibition.

logging.info("Cross Validation")


def f_multi(x):
    process, alpha, no_grid = x
    return fit_model.cross_validate(
        dfs_dense[no_grid],
        data_df,
        process,
        f_spline,
        odes[alpha][no_grid],
        Y_XG,
        Y_PG,
        alpha,
    )


with multiprocessing.Pool(100) as p:
    CV_results = p.map(f_multi, itertools.product(process_names, alphas, no_points))

res_CV = dict()
i = 0
for p in process_names:
    res_CV[p] = dict()
    for a in alphas:
        res_CV[p][a] = dict()
        for n in no_points:
            res_CV[p][a][n] = CV_results[i]
            i += 1

# CV_results have the form [(dfs_var: dict containing parameter selection results,
# df_CV: results df for measurementpoints), df_CV_plot: results df with more points for plots]
# ode_CV: instance of Ode object (ode_CV.f ist the ODE in the form f(x, u, x0))

with open(f"{data_path}CV_results.pickle", "wb") as f:
    pickle.dump(res_CV, f)


def dict_list():
    return collections.defaultdict(list)


def dict_dict():
    return collections.defaultdict(dict)


dfs_CV = collections.defaultdict(dict_list)
dfs_CV_plot = collections.defaultdict(dict_list)
odes_CV = collections.defaultdict(dict_dict)

for (_, df, df_plot, ode), x in zip(
    CV_results, itertools.product(process_names, alphas, no_points)
):
    process_name, alpha, no_points = x
    dfs_CV[alpha][no_points].append(df)
    dfs_CV_plot[alpha][no_points].append(df_plot)
    odes_CV[alpha][no_points][process_name] = ode

for alpha, a in dfs_CV.items():
    for no_points, b in a.items():
        dfs_CV[alpha][no_points] = pd.concat(b)

for alpha, a in dfs_CV_plot.items():
    for no_points, b in a.items():
        dfs_CV_plot[alpha][no_points] = pd.concat(b)


with open(f"{data_path}dfs_CV.pickle", "wb") as f:
    pickle.dump(dfs_CV, f)

with open(f"{data_path}dfs_CV_plot.pickle", "wb") as f:
    pickle.dump(dfs_CV_plot, f)

with open(f"{data_path}odes_CV.pickle", "wb") as f:
    pickle.dump(odes_CV, f)

logging.info("finished cross validation")

# %%

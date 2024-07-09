# %%
import numpy as np
import pandas as pd

# import read_data
import test_model
import fit_model
import importlib
import multiprocessing
import collections
import pickle
import calc_rates
import logging
import itertools

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

data_dir = "../data/"

# yields
Y_XG = 0.627
Y_PG = 0.652
c_feed = 390

# we reate addititonal_parameters * process_per_paremeter processes
additional_paramesters = range(16)
process_per_parameter = 25
P_min = 3  # models not reaching this value in all DoE runs are discarded

# set values for random errors here:
P_errors = np.array((0, 0.02, 0.04, 0.08, 0.16, 0.32))
X_errors = np.array((0, 0.015, 0.03, 0.06, 0.12))
G_errors = np.array([0])
X_errors_abs = np.array([0])
P_errors_abs = np.array([0])
G_errors_abs = np.array([0])

# in total addititonal_parameters * process_per_paremeter * len(P_errors) * len(X_errors) models are fitted (25 * 16 * 6 * 5 = 12000)


# %%
def calc_DoE(mu0, dmu, T0, dT):
    return [[mu0, T0]] * 4 + [
        [mu0 + dmu, T0 + dT],
        [mu0 + dmu, T0 - dT],
        [mu0 - dmu, T0 + dT],
        [mu0 - dmu, T0 - dT],
        [mu0, T0 + dT * 2**0.5],
        [mu0, T0 - dT * 2**0.5],
        [mu0 + dmu * 2**0.5, T0],
        [mu0 - dmu * 2**0.5, T0],
    ]


DoE = calc_DoE(0.12, 0.08, 304.15, 4)

with open(f"{data_dir}DoE.pickle", "wb") as f:
    pickle.dump(DoE, f)


# get maximal measured values to get realistic simulated models

process_names = [
    "DoE1_R2",
    "DoE1_R4",
    "DoE2_R2",
    "DoE2_R4",
    "DoE3_R1",
    "DoE1_R3",
    "DoE2_R3",
    "DoE3_R3",
    "DoE3_R4",
    "DoE3_R2",
    "DoE2_R1",
    "DoE1_R1",
]

data_df = pd.read_csv(f"{data_dir}sampling_points.csv", index_col=0)
df_Vf = pd.read_csv(f"{data_dir}volume_flow_rates.csv", index_col=0)
df_T = pd.read_csv(f"{data_dir}temperatures.csv", index_col=0)

with open(f"{data_dir}c_feed.pickle", "rb") as f:
    c_feeds_training = pickle.load(f)

df_measurement_points, df_dense, f_spline = calc_rates.calc_rates_df(
    data_df,
    df_Vf,
    c_feeds_training,
    Y_XG,
    Y_PG,
    df_T=df_T,
    k=3,
    s=10,
)

G_max = np.max(df_measurement_points.G)
n_max = np.max(df_measurement_points.n)
PX_max = np.max(df_measurement_points.P)
X_max = np.max(df_measurement_points.X)

c_feeds = dict(zip([f"p{i}" for i in range(17)], np.repeat(c_feed, 17)))

# X0 = np.mean(data_df[data_df.t == 0].X / data_df[data_df.t == 0].V)
X0 = 30

logging.info("create random models")

test_processes_exact = dict()
for no_params in additional_paramesters:
    res = list()
    logging.info(f"{no_params} additional parmeter")
    j = 0
    k = 0
    for i in range(process_per_parameter):
        P_max = 0
        while P_max < P_min:
            params = test_model.create_params(no_params, G_max, n_max, PX_max, X_max)
            df, df_Vf = test_model.sim_processes(
                params,
                X0,
                Y_XG,
                Y_PG,
                c_feed,
                us=DoE,
                ode_solver="LSODA",
                ode_options={"min_step": 1e-12},
            )
            j += 1
            if df is None:  # remove parameter where numeric integration fails
                continue

            if j % 1000 == 0:
                logging.info(f"{j} processes created, {len(res)} accepted")
            P_max = df.P.max()

        res.append([params, df, df_Vf])
    test_processes_exact[no_params] = res
    logging.info(f"models created: {j}")


logging.info("add random errors")


def add_errors(
    test_processes_exact: dict[int, list],
    errors: tuple,
):
    """adds errors, if more than 1 set of errors is given, creates data for each
    test process and each set of errors.
    errors given as:
    ((rel_error_X, rel_error_P, rel_error_G, abs_error_X, abs_error_P, abs_error_G), ...)
    """
    test_processes = dict()
    for error in errors:
        X_error, P_error, G_error, X_error_abs, P_error_abs, G_error_abs = error
        res_error = dict()
        for no_params, dfs in test_processes_exact.items():
            res_params = list()
            for params, df, df_Vf in dfs:
                df_err = test_model.add_errors(
                    df,
                    sigma_G_abs=G_error_abs,
                    sigma_G_rel=G_error,
                    sigma_X_abs=X_error_abs,
                    sigma_X_rel=X_error,
                    sigma_P_abs=P_error_abs,
                    sigma_P_rel=P_error,
                )
                res_params.append([params, df_err, df_Vf])
            res_error[no_params] = res_params

        test_processes[error] = res_error
    return test_processes


errors = tuple(
    itertools.product(
        X_errors, P_errors, G_errors, X_errors_abs, P_errors_abs, G_errors_abs
    )
)

test_processes = add_errors(test_processes_exact, errors)

test_processes_list = list()
for error, params in test_processes.items():
    for no_params, dfs in params.items():
        for i, (p, df, df_Vf) in enumerate(dfs):
            test_processes_list.append([error, no_params, i, p, df, df_Vf])

# pickle test_process exact
with open(f"{data_dir}test_process_exact.pickle", "wb") as f:
    pickle.dump(test_processes_exact, f)

with open(f"{data_dir}test_processes_list.pickle", "wb") as f:
    pickle.dump(test_processes_list, f)

# fit models using our framework, this is done for all processes and error levels

logging.info("fitting model using the framework")


def fit(input):
    error, no_params, i, p, df, df_Vf = input
    df_measurement_points, _, _ = calc_rates.calc_rates_df(
        df, df_Vf, c_feeds, Y_XG, Y_PG, k=3, s=10
    )
    return fit_model.fit_model(df_measurement_points, Y_XG, Y_PG, alpha=0.2)


with multiprocessing.Pool(100) as p:
    res_fit = p.map(fit, test_processes_list)

process_list0 = [a + list(b) for a, b in zip(test_processes_list, res_fit)]

importlib.reload(test_model)

process_list = [
    x + list(test_model.compare_estimates_with_real_data(x[3], x[7]))
    for x in process_list0
]

logging.info("saving results")

with open(f"{data_dir}val_results.pickle", "wb") as f:
    pickle.dump(process_list, f)


with open(f"{data_dir}val_results.pickle", "wb") as f:
    pickle.dump(process_list, f)

with open(f"{data_dir}val_processes.pickle", "wb") as f:
    pickle.dump(test_processes_exact, f)

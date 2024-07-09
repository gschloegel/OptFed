import pickle
import matplotlib.colors

with open("../data/moma_colors.pickle", "rb") as f:
    moma_colors = pickle.load(f)

# for processes 3 for training data, 2 for optima

processes = moma_colors["vonHeyl"]
# processes = moma_colors["Ohchi"]


# for abstract optima( 2 for longer processer)

opt_longer_processes = moma_colors["VanGogh"][:2]

# plus minus (2 colors, colormap)

minus = matplotlib.colors.to_rgba("xkcd:blue")
plus = matplotlib.colors.to_rgba("xkcd:red")
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", [minus, (1, 1, 1, 1), plus]
)

# genaral plots

general = moma_colors["Palermo"]

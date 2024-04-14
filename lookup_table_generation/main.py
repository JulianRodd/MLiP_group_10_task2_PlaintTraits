import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_csv = pd.read_csv("train.csv")
mean_columns = [
    col
    for col in train_csv.columns
    if "mean" in col
    and all(x not in col for x in ["WORLDCLIM_BIO", "SOIL", "MODIS", "VOD", "sd"])
]


def group_rows(df, mean_columns, min_matches, precision, save_csv=False):
    scaling_params = {}
    for col in mean_columns:
        max_val = df[col].max()
        min_val = df[col].min()
        scaling_params[col] = (min_val, max_val)
        df[col] = (df[col] - min_val) / (max_val - min_val)
        df[col] = df[col].round(precision)

    df["group_key"] = df[mean_columns].astype(str).agg("-".join, axis=1)
    grouped = df.groupby("group_key").groups
    valid_groups = {
        key: grouped[key] for key in grouped if len(grouped[key]) >= min_matches
    }

    group_codes = {key: i + 1 for i, key in enumerate(valid_groups.keys())}
    df["group_code"] = df["group_key"].map(group_codes).fillna(0)

    for col in mean_columns:
        min_val, max_val = scaling_params[col]
        df[col] = df[col] * (max_val - min_val) + min_val

    df.drop(columns=["group_key"], inplace=True)

    lookup = (
        df[["group_code"] + mean_columns].groupby("group_code").mean().reset_index()
    )
    if save_csv:

        df.to_csv("grouped_train.csv", index=False)

        lookup.to_csv("lookup.csv", index=False)

        with open("params.txt", "w") as f:
            f.write(f"min_matches: {min_matches}\n")
            f.write(f"precision: {precision}\n")
    return lookup


def run_varied_params():

    results = []
    for precision in range(1, 16):
        for min_matches in range(4, 7):
            lookup = group_rows(train_csv.copy(), mean_columns, min_matches, precision)
            results.append((precision, min_matches, len(lookup)))

    return results


def plot_results(results, optimal_count=200):

    precisions, min_matches, counts = zip(*results)
    precisions = np.array(precisions)
    min_matches = np.array(min_matches)
    counts = np.array(counts)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        precisions,
        min_matches,
        c=counts,
        cmap="viridis",
        s=100,
        edgecolor="k",
        alpha=0.6,
    )
    plt.colorbar(scatter, label="Number of Groups in Lookup Table")

    plt.xlabel("Precision")
    plt.ylabel("Minimum Matches")
    plt.title("Distribution of Group Counts in Lookup Table")

    plt.yticks(np.arange(min(min_matches), max(min_matches) + 1, 1))
    plt.ylim(min(min_matches) - 0.5, max(min_matches) + 0.5)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    optimal_index = np.abs(counts - optimal_count).argmin()
    optimal_precision = precisions[optimal_index]
    optimal_min_matches = min_matches[optimal_index]
    plt.scatter(
        [optimal_precision],
        [optimal_min_matches],
        color="red",
        s=150,
        label=f"Optimal ({optimal_count} Groups)",
        edgecolors="black",
    )
    plt.legend()

    plt.savefig("images/lookup_distribution.png")

    optimal_params = {
        "precision": optimal_precision,
        "min_matches": optimal_min_matches,
    }
    return optimal_params


results = run_varied_params()
optimal_params = plot_results(results, optimal_count=300)

optimal_lookup = group_rows(
    pd.read_csv("train.csv"),
    mean_columns,
    optimal_params["min_matches"],
    optimal_params["precision"],
    save_csv=True,
)

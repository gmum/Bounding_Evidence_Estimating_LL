#!/usr/bin/env python
# coding: utf-8
import argparse
from math import log
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import yaml
from scipy.special import logsumexp, softmax
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Runner for VAE model")
parser.add_argument(
    "--dataname", type=str, choices=["MNIST", "SVHN", "CelebA"], default="MNIST"
)
parser.add_argument(
    "--modelname", type=str, choices=["VAE", "IWAE-5", "IWAE-10"], default="VAE"
)
parser.add_argument("--num_files", type=int, default=10)
parser.add_argument(
    "--resume_name",
    choices=["train_upper_bound", "train_gap", "test_upper_bound", "test_gap"],
    default=None,
)
parser.add_argument("--save2file", type=str, default="scores.csv")
parser.add_argument("--num_feature", type=int, nargs="+", default=[1024])
args = parser.parse_args()
print(args)
args_dict = vars(args)

filenames = {
    "MNIST": {
        "VAE": "./results/VAEmnist_${???}/config.yaml",  # todo: type date instead ${???} e.g. 2022-01-15_193505
        "IWAE-5": "./results/VAEmnist_${???}/config.yaml",  # todo: type date instead ${???} e.g. 2022-01-15_193505
        "IWAE-10": "./results/VAEmnist_${???}/config.yaml",  # todo: type date instead ${???} e.g. 2022-01-15_193505
    },
    "SVHN": {
        "VAE": "./results/VAEsvhn_${???}/config.yaml",  # todo: type date instead ${???} e.g. 2022-01-15_193505
        "IWAE-5": "./results/VAEsvhn_${???}/config.yaml",  # todo: type date instead ${???} e.g. 2022-01-15_193505
        "IWAE-10": "./results/VAEsvhn_${???}/config.yaml",  # todo: type date instead ${???} e.g. 2022-01-15_193505
    },
    "CelebA": {
        "VAE": "./results/VAEceleba_${???}/config.yaml",  # todo: type date instead ${???} e.g. 2022-01-15_193505
        "IWAE-5": "./results/VAEceleba_${???}/config.yaml",  # todo: type date instead ${???} e.g. 2022-01-15_193505
        "IWAE-10": "./results/VAEceleba_${???}/config.yaml",  # todo: type date instead ${???} e.g. 2022-01-15_193505
    },
}

filename = filenames[args.dataname][args.modelname]

with open(filename, "r") as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

pprint(config)
filename = str(Path(filename).parent)

scores = {dim: {} for dim in args.num_feature}

N_SAMPLES = None
num_files = 1

values2bound_I = values2bound_II = None
max_feature = np.max(args.num_feature)

files = list(Path(f"{filename}/values").glob("**/*.npz"))
bar_tqdm = tqdm(files, total=len(files))
for file in bar_tqdm:

    data = np.load(file)
    #     print(file, data.files)
    TEMP_values2bound_I, TEMP_values2bound_II = map(
        lambda x: data[x], ("values2bound_I", "values2bound_II")
    )

    if N_SAMPLES is None:
        N_SAMPLES = TEMP_values2bound_I.shape[0]

    if values2bound_I is None:
        values2bound_I = TEMP_values2bound_I
        values2bound_II = TEMP_values2bound_II
    else:
        values2bound_I = np.concatenate([values2bound_I, TEMP_values2bound_I], axis=1)
        values2bound_II = np.concatenate(
            [values2bound_II, TEMP_values2bound_II], axis=1
        )
    if values2bound_I.shape[1] < max_feature:
        continue

    values2bound_I, values2bound_II = map(
        lambda x: x[:, :max_feature], (values2bound_I, values2bound_II)
    )

    assert (
        values2bound_I.shape[1] == values2bound_II.shape[1] == max_feature
    ), f"{values2bound_I.shape[1]}|{values2bound_II.shape[1]}"
    assert (
        values2bound_I.shape[0] == values2bound_II.shape[0] == N_SAMPLES
    ), f"{values2bound_I.shape[0]}|{values2bound_II.shape[0]}"

    for dim in tqdm(args.num_feature, leave=False, desc="Dimensional"):
        data1, data2 = map(lambda x: x[:, :dim], (values2bound_I, values2bound_II))

        log_X, log_Y = map(
            lambda x: logsumexp(x, axis=1) - log(float(x.shape[1])),
            (data1, data2),
        )

        try:
            scores[dim]["log_DIV"] = np.vstack([scores[dim]["log_DIV"], log_Y - log_X])
        except KeyError:
            scores[dim]["log_DIV"] = log_Y - log_X

        try:
            scores[dim]["ELBO"] = np.vstack([scores[dim]["ELBO"], log_X])
        except KeyError:
            scores[dim]["ELBO"] = log_X

        for n in [1.5, 2]:
            cubo = (logsumexp(n * data1, axis=1) - log(float(data1.shape[1]))) / n
            try:
                scores[dim][f"CUBO_{n}"] = np.vstack([scores[dim][f"CUBO_{n}"], cubo])
            except KeyError:
                scores[dim][f"CUBO_{n}"] = cubo

        for K in [1, 2, 5, 10, 50]:
            TVO = ELBO_TVO = 0
            for k in range(0, K + 1):
                beta = k / K
                log_w_z = beta * data1
                w_z = softmax(log_w_z, axis=1)
                tmp = np.sum(w_z * data1, axis=1)
                if k < K:
                    ELBO_TVO = ELBO_TVO + tmp
                if k > 0:
                    TVO = TVO + tmp
            TVO /= K
            ELBO_TVO /= K
            try:
                scores[dim][f"TVO_{K}"] = np.vstack([scores[dim][f"TVO_{K}"], TVO])
                scores[dim][f"ELBO_TVO_{K}"] = np.vstack(
                    [scores[dim][f"ELBO_TVO_{K}"], ELBO_TVO]
                )
            except KeyError:
                scores[dim][f"TVO_{K}"] = TVO
                scores[dim][f"ELBO_TVO_{K}"] = ELBO_TVO

    values2bound_I = values2bound_II = None

    if num_files == args.num_files:
        break
    num_files += 1
    bar_tqdm.set_description(f"Files: {num_files}")

print(f"\033[0;1;32m{num_files=}\033[0m")

del args_dict["resume_name"], args_dict["num_feature"], args_dict["save2file"]

df = None
if Path(args.save2file).is_file():
    df = pd.read_csv(args.save2file)

for key, val in args_dict.items():
    args_dict[key] = [val]

for dim in args.num_feature:
    logEX_Y = logsumexp(scores[dim]["log_DIV"], axis=0) - log(
        scores[dim]["log_DIV"].shape[0]
    )  # logE(Y/X)
    gap = logEX_Y

    tab = {}
    for key, val in scores[dim].items():
        if key == "log_DIV":
            continue

        if key == "ELBO":
            our_ = val + gap[np.newaxis, ...]
            tab["our"] = {
                "mean": np.mean(our_, axis=0),
                "std": np.std(our_, axis=0),
            }
            for sub_key in ["std", "mean"]:
                tab["our"][sub_key] = np.mean(tab["our"][sub_key])

        tab[key] = {"mean": np.mean(val, axis=0), "std": np.std(val, axis=0)}
        for sub_key in ["std", "mean"]:
            tab[key][sub_key] = np.mean(tab[key][sub_key])

    for key in tab.keys():
        if key.startswith("ELBO"):
            continue
        if key.startswith(("our", "CUBO")):
            tab[key]["gap"] = tab[key]["mean"] - tab["ELBO"]["mean"]
        else:
            tab[key]["gap"] = tab[key]["mean"] - tab[f"ELBO_{key}"]["mean"]

    args_dict["feature"] = dim
    df_temp = pd.DataFrame.from_dict(tab)
    for stat in ["mean", "std", "gap"]:
        args_dict.update(
            {
                f"{key}-{stat}": val
                for key, val in df_temp.loc[stat].to_dict().items()
                if np.isfinite(val)
            }
        )

    if df is None:
        df = pd.DataFrame.from_dict(args_dict)
    else:
        df = pd.concat(
            [df, pd.DataFrame.from_dict(args_dict)],
            axis=0,
            join="outer",
            ignore_index=False,
        )

df.to_csv(args.save2file, index=False)

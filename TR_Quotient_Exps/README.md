# TR quotient experiments

This directory contains the runnable scripts for the numerical experiments in
“Quotient geometry of tensor ring decomposition.” Generated MAT, EPS, and TEX
files are written to `Results/`.

| Experiment | Run script | Plot script |
|---|---|---|
| Different geometries | `Exp1_different_geometries.m` | `Exp1_different_geometries_results.m` |
| Higher-order and scaling tests | `Exp2_scaling.m` | `Exp2_scaling_results.m` |
| Sample complexity | `Exp3_phase_TR.m`, `Exp3_phase_uTR.m` | `Exp3_phase_results.m` |
| MovieLens 1M | `Exp4_movielens.m` | `Exp4_movielens_results.m` |

`Exp1_run_TR_E.m`, `Exp1_run_uTR_E.m`, `Exp2_run_E.m`, and the MovieLens
helper files implement algorithm- or dataset-specific operations used by the
scripts.

## MovieLens 1M data

Experiment 4 requires the MovieLens 1M `ratings.dat` file. The dataset is not
included in this repository because the GroupLens license does not permit
redistribution without separate permission.

Download `ml-1m.zip` from the
[official GroupLens page](https://grouplens.org/datasets/movielens/1m/),
extract it, and copy `ratings.dat` into this directory before running
`Exp4_movielens.m`.

Please acknowledge the dataset using:

F. Maxwell Harper and Joseph A. Konstan, “The MovieLens Datasets: History and
Context,” *ACM Transactions on Interactive Intelligent Systems*, 5(4),
Article 19, 2015. https://doi.org/10.1145/2827872

## Run

Open MATLAB and run an experiment script directly, for example:

```matlab
cd('/path/to/LRTCTR/TR_Quotient_Exps')
Exp1_different_geometries
Exp2_scaling
Exp3_phase_TR
Exp3_phase_uTR
Exp4_movielens
```

Each script loads the parent LRTCTR package through `install.m`. Run the
matching `*_results.m` script separately when only an existing MAT result
needs to be plotted. For Experiment 2, setting `LRTCTR_EXPS_LOCAL=1` selects
the intermediate-size configuration, and `LRTCTR_EXPS_PLOT_ONLY=1` plots an
existing result without rerunning the experiment.

The phase-plot commands in `Exp3_phase_results.m` are intentionally commented
out. Uncomment them only when those plots are needed.

The scripts use the current public solver interfaces and evaluate sampled TR
entries directly; they do not form a full dense tensor.

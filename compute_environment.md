# Compute Environment and Search-Time Reporting

This file documents the hardware, parallelism settings, and per-model/per-method
search cost used to produce the results in `resultados_todos.csv`, in response to
the reviewer's request to report CPU model, core count, `n_jobs`, number of
candidate configurations, and mean time per trial for every search method.

## Hardware

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 5700G with Radeon Graphics, 3.80 GHz base |
| Cores / Threads | 8 cores / 16 threads (Zen 3) |
| RAM | 16.0 GB (15.4 GB usable) |
| OS | Windows, 64-bit |

## `n_jobs` configuration per search method (from source code)

| Search method | `n_jobs` at the searcher level | Behavior |
|---|---|---|
| Grid Search (`GridSearchCV`) | `n_jobs=-1` | Candidate × fold fits are distributed across all available threads. |
| Random Search (`RandomizedSearchCV`) | `n_jobs=-1` | Same as above. |
| Optuna (`study.optimize`) | not set (Optuna default = sequential, 1 trial at a time) | Trials are evaluated one after another; the inner 5×3 cross-validation loop inside each objective function is a plain Python `for` loop, not parallelized. |

Additionally, for tree-ensemble models (`RandomForestClassifier`, `XGBClassifier`,
`LGBMClassifier`, `BaggingClassifier`) the estimator itself is instantiated with
`n_jobs=-1` regardless of search method, so a single model fit can use multiple
threads on its own. Logistic Regression, LinearSVC, and RidgeClassifier have no
such internal multithreading.

Net effect: Grid Search and Random Search parallelize **across candidate
configurations**, while Optuna parallelizes **only within a single model fit**
(for tree ensembles) and evaluates configurations strictly one at a time.

## Candidate configurations and search time per model/method

Values below are taken directly from `resultados_todos.csv`
(`n_evaluaciones`, `tiempo_busqueda_seg`); `mean_time_per_trial_s` is computed
as `tiempo_busqueda_seg / n_evaluaciones`.

| Model | Method | Candidates evaluated | Total search time (s) | Mean time / trial (s) |
|---|---|---:|---:|---:|
| Regresion Logistica | Grid Search | 60 | 51.16 | 0.853 |
| Regresion Logistica | Random Search | 200 | 44.14 | 0.221 |
| Regresion Logistica | Optuna | 200 | 331.13 | 1.656 |
| LinearSVC | Grid Search | 15 | 1.40 | 0.093 |
| LinearSVC | Random Search | 200 | 4.00 | 0.020 |
| LinearSVC | Optuna | 200 | 21.51 | 0.108 |
| RidgeClassifier | Grid Search | 15 | 0.36 | 0.024 |
| RidgeClassifier | Random Search | 200 | 3.88 | 0.019 |
| RidgeClassifier | Optuna | 200 | 21.57 | 0.108 |
| Random Forest | Grid Search | 162 | 166.79 | 1.030 |
| Random Forest | Random Search | 200 | 195.09 | 0.975 |
| Random Forest | Optuna | 200 | 1399.38 | 6.997 |
| XGBoost | Grid Search | 162 | 228.67 | 1.412 |
| XGBoost | Random Search | 200 | 34.33 | 0.172 |
| XGBoost | Optuna | 200 | 242.58 | 1.213 |
| LightGBM | Grid Search | 162 | 190.75 | 1.177 |
| LightGBM | Random Search | 200 | 168.69 | 0.843 |
| LightGBM | Optuna | 200 | 104.70 | 0.523 |
| Bagging | Grid Search | 81 | 98.06 | 1.211 |
| Bagging | Random Search | 200 | 190.44 | 0.952 |
| Bagging | Optuna | 200 | 344.03 | 1.720 |

## Note on the Random Forest case (1399 s vs 195 s)

Random Forest is the model with the largest gap: Optuna took 1399.38 s for 200
trials (6.997 s/trial) versus 195.09 s for Random Search's 200 trials
(0.975 s/trial) — a ~7.2x ratio. Since both methods evaluated the same number of
candidates (200), this ratio is a mean-time-per-trial effect, not a difference in
sample size. It is consistent with (a) `RandomizedSearchCV`'s `n_jobs=-1`
distributing the 200 candidates × 15 inner folds across up to 16 threads at once,
while Optuna evaluated the same 3,000 fold-fits fully sequentially, and (b)
whatever `n_estimators` values were actually sampled by each method (reported per
model in `resultados_todos.csv` under `best_params`, though that column only
shows the best trial, not the full sampled distribution).

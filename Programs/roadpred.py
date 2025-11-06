# Import libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")

# Set seeds for reproducibility
SEEDS = [42, 181, 2025]
np.random.seed(SEEDS[0])

# File paths
INPUT_DIR = "/kaggle/input/playground-series-s5e10"
SYN_DIR = "/kaggle/input/simulated-roads-accident-data"
TRAIN_P = os.path.join(INPUT_DIR, "train.csv")
TEST_P = os.path.join(INPUT_DIR, "test.csv")
SAMP_P = os.path.join(INPUT_DIR, "sample_submission.csv")

# Load data
df_train = pd.read_csv(TRAIN_P)
df_test = pd.read_csv(TEST_P)
df_sample = pd.read_csv(SAMP_P)

# Load synthetic data
syn_paths = [
    os.path.join(SYN_DIR, f"synthetic_road_accidents_{s}k.csv") for s in [2, 10, 100]
    if os.path.exists(os.path.join(SYN_DIR, f"synthetic_road_accidents_{s}k.csv"))
]
df_syn = pd.concat([pd.read_csv(p) for p in syn_paths], axis=0, ignore_index=True) if syn_paths else pd.DataFrame()

# Data preparation
target_col = "accident_risk"
if target_col not in df_test.columns:
    df_test[target_col] = 0.5

n_train = len(df_train)
n_test = len(df_test)

# Add IDs to synthetic data
if not df_syn.empty:
    if "id" not in df_syn.columns:
        start_id = int(df_test["id"].max()) + 1
        df_syn.insert(0, "id", np.arange(start_id, start_id + len(df_syn)))
    for c in df_train.columns:
        if c not in df_syn.columns:
            df_syn[c] = np.nan
    df_syn = df_syn[df_train.columns]

# Combine datasets
df_all = pd.concat([df_train, df_test, df_syn], axis=0, ignore_index=True)
print("Combined shape:", df_all.shape)

# Convert boolean to int
for c in df_all.select_dtypes(include="bool").columns:
    df_all[c] = df_all[c].astype(int)

# Convert object to string and strip
for c in df_all.select_dtypes(include="object").columns:
    df_all[c] = df_all[c].astype(str).str.strip()

# Baseline risk function
def road_risk(X):
    return (
        0.3 * X["curvature"] +
        0.2 * (X["lighting"] == "night").astype(int) +
        0.1 * (X["weather"] != "clear").astype(int) +
        0.2 * (X["speed_limit"] >= 60).astype(int) +
        0.1 * (X["num_reported_accidents"] > 2).astype(int)
    )

# Clipped risk function
def clipped(func):
    def clip_f(X):
        mu = func(X)
        sigma = 0.05
        a, b = -mu / sigma, (1 - mu) / sigma
        Phi_a, Phi_b = norm.cdf(a), norm.cdf(b)
        phi_a, phi_b = norm.pdf(a), norm.pdf(b)
        return mu * (Phi_b - Phi_a) + sigma * (phi_a - phi_b) + 1 - Phi_b
    return clip_f

df_all["y"] = clipped(road_risk)(df_all)

# Advanced feature engineering
print("Starting advanced feature engineering...")

# Interaction features
df_all['road_weather'] = df_all['road_type'].astype(str) + '_' + df_all['weather'].astype(str)
df_all['road_light'] = df_all['road_type'].astype(str) + '_' + df_all['lighting'].astype(str)
df_all['weather_light'] = df_all['weather'].astype(str) + '_' + df_all['lighting'].astype(str)
df_all['curvature_lighting'] = df_all['curvature'] * df_all['lighting'].map({'daylight': 0, 'dim': 1, 'night': 2})
df_all['curvature_weather'] = df_all['curvature'] * df_all['weather'].map({'clear': 0, 'rainy': 1, 'foggy': 2})
df_all['speed_weather'] = df_all['speed_limit'] * df_all['weather'].map({'clear': 0, 'rainy': 1, 'foggy': 2})

# Polynomial features
df_all['curvature_sq'] = df_all['curvature'] ** 2
df_all['speed_limit_sq'] = df_all['speed_limit'] ** 2

# Aggregated features
mean_risk_road_type = df_train.groupby('road_type')[target_col].mean().to_dict()
df_all['mean_risk_road_type'] = df_all['road_type'].map(mean_risk_road_type)
mean_risk_time = df_train.groupby('time_of_day')[target_col].mean().to_dict()
df_all['mean_risk_time'] = df_all['time_of_day'].map(mean_risk_time)

# Log transformation
df_all['log_curvature'] = np.log1p(df_all['curvature'])

new_interaction_features = [
    'road_weather', 'road_light', 'weather_light',
    'curvature_lighting', 'curvature_weather', 'speed_weather',
    'curvature_sq', 'speed_limit_sq', 'mean_risk_road_type',
    'mean_risk_time', 'log_curvature'
]
print(f"Added {len(new_interaction_features)} new features.")

# Categorize features
original_cols = [col for col in df_train.columns if col not in ['id', target_col]]
CATS, NUMS = [], []
CATS.extend(new_interaction_features[:3])  # road_weather, road_light, weather_light
for col in original_cols:
    if df_all[col].dtype == "object":
        CATS.append(col)
    else:
        NUMS.append(col)
CATS = list(dict.fromkeys(CATS))

# Factorize categorical columns
print(f"Factorizing {len(CATS)} categorical columns: {CATS}")
for col in CATS:
    df_all[col], _ = df_all[col].factorize()

# Add "y" to features
FEATURES = CATS + NUMS + ['y'] + new_interaction_features[3:]

# Split back into train, test, synthetic
df_train_p = df_all.iloc[:n_train].reset_index(drop=True)
df_test_p = df_all.iloc[n_train:n_train + n_test].reset_index(drop=True)
df_syn_p = df_all.iloc[n_train + n_test:].reset_index(drop=True) if not df_syn.empty else pd.DataFrame()
print("Sizes -> train:", len(df_train_p), "test:", len(df_test_p), "synthetic:", len(df_syn_p))

# Target encoding
TE_features = []
te_source = pd.concat([df_train_p, df_syn_p], axis=0, ignore_index=True)
print(f"Using {len(FEATURES)} features for Target Encoding...")
for col in FEATURES:
    te_map = te_source.groupby(col)[target_col].mean()
    te_name = f"TE_{col}"
    df_train_p[te_name] = df_train_p[col].map(te_map)
    df_test_p[te_name] = df_test_p[col].map(te_map)
    TE_features.append(te_name)
print("Target Encoding complete.")

# Final feature set
FINAL_FEATURES = FEATURES + TE_features
print("Final feature count:", len(FINAL_FEATURES))

# Initialize storage for predictions
FOLDS = 7
all_oof_preds = {model: np.zeros((len(SEEDS), len(df_train_p))) for model in ['xgb', 'lgb', 'cat', 'mlp']}
all_test_preds = {model: np.zeros((len(SEEDS), len(df_test_p))) for model in ['xgb', 'lgb', 'cat', 'mlp']}

# Model parameters
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.01,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.6,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": -1,
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "verbose": -1,
}

cat_params = {
    "loss_function": "RMSE",
    "iterations": 10000,
    "learning_rate": 0.01,
    "depth": 6,
    "subsample": 0.8,
    "colsample_bylevel": 0.6,
    "verbose": 0,
    "early_stopping_rounds": 200,
}

# MLP model
def create_mlp(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Training loop
for seed_idx, seed in enumerate(SEEDS):
    print(f"\n--- Training with SEED: {seed} ({seed_idx + 1}/{len(SEEDS)}) ---")
    np.random.seed(seed)
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)

    for model_name in ['xgb', 'lgb', 'cat', 'mlp']:
        oof_preds_seed = np.zeros(len(df_train_p))
        test_preds_seed = np.zeros(len(df_test_p))

        print(f"Training {model_name.upper()} with {FOLDS}-fold CV...")
        for fold, (tr_idx, val_idx) in enumerate(kf.split(df_train_p), 1):
            print(f"  Fold {fold}/{FOLDS}", end=" ... ")
            X_tr = df_train_p.iloc[tr_idx][FINAL_FEATURES]
            X_val = df_train_p.iloc[val_idx][FINAL_FEATURES]
            y_tr = df_train_p.iloc[tr_idx][target_col].values - df_train_p.iloc[tr_idx]["y"].values
            y_val = df_train_p.iloc[val_idx][target_col].values - df_train_p.iloc[val_idx]["y"].values

            if model_name == 'xgb':
                dtrain = xgb.DMatrix(X_tr, label=y_tr)
                dval = xgb.DMatrix(X_val, label=y_val)
                dtest = xgb.DMatrix(df_test_p[FINAL_FEATURES])
                model = xgb.train(
                    params={**xgb_params, "seed": seed},
                    dtrain=dtrain,
                    num_boost_round=100000,
                    evals=[(dtrain, "train"), (dval, "valid")],
                    early_stopping_rounds=200,
                    verbose_eval=False
                )
                val_pred = model.predict(dval) + df_train_p.iloc[val_idx]["y"].values
                test_pred = model.predict(dtest) + df_test_p["y"].values

            elif model_name == 'lgb':
                dtrain = lgb.Dataset(X_tr, label=y_tr)
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
                model = lgb.train(
                    params={**lgb_params, "seed": seed},
                    train_set=dtrain,
                    valid_sets=[dtrain, dval],
                    num_boost_round=10000,
                    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
                )
                val_pred = model.predict(X_val) + df_train_p.iloc[val_idx]["y"].values
                test_pred = model.predict(df_test_p[FINAL_FEATURES]) + df_test_p["y"].values

            elif model_name == 'cat':
                dtrain = cb.Pool(X_tr, y_tr)
                dval = cb.Pool(X_val, y_val)
                model = cb.CatBoostRegressor(**cat_params, random_seed=seed)
                model.fit(dtrain, eval_set=dval, use_best_model=True)
                val_pred = model.predict(X_val) + df_train_p.iloc[val_idx]["y"].values
                test_pred = model.predict(df_test_p[FINAL_FEATURES]) + df_test_p["y"].values

            elif model_name == 'mlp':
                model = create_mlp(len(FINAL_FEATURES))
                model.fit(X_tr, y_tr, epochs=50, batch_size=32, verbose=0)
                val_pred = model.predict(X_val, verbose=0).flatten() + df_train_p.iloc[val_idx]["y"].values
                test_pred = model.predict(df_test_p[FINAL_FEATURES], verbose=0).flatten() + df_test_p["y"].values

            oof_preds_seed[val_idx] = np.clip(val_pred, 0, 1)
            test_preds_seed += np.clip(test_pred, 0, 1) / FOLDS
            print("done")

        all_oof_preds[model_name][seed_idx] = oof_preds_seed
        all_test_preds[model_name][seed_idx] = test_preds_seed
        rmse_oof_seed = np.sqrt(mean_squared_error(df_train_p[target_col], oof_preds_seed))
        print(f"SEED {seed} {model_name.upper()} OOF RMSE: {rmse_oof_seed:.5f}")

# Average predictions across seeds
final_oof_preds = {}
final_test_preds = {}
for model_name in all_oof_preds:
    final_oof_preds[model_name] = np.mean(all_oof_preds[model_name], axis=0)
    final_test_preds[model_name] = np.mean(all_test_preds[model_name], axis=0)
    rmse_oof = np.sqrt(mean_squared_error(df_train_p[target_col], final_oof_preds[model_name]))
    print(f"Final {model_name.upper()} OOF RMSE: {rmse_oof:.5f}")

# Stacking
print("\nTraining stacking meta-model...")
meta_X = np.column_stack([final_oof_preds[model] for model in ['xgb', 'lgb', 'cat', 'mlp']])
meta_model = Ridge()
meta_model.fit(meta_X, df_train_p[target_col])
meta_test = np.column_stack([final_test_preds[model] for model in ['xgb', 'lgb', 'cat', 'mlp']])
stacked_preds = meta_model.predict(meta_test)

# Calibration
print("Calibrating predictions...")
ir = IsotonicRegression(out_of_bounds="clip")
ir.fit(final_oof_preds['xgb'], df_train_p[target_col])  # Use XGBoost as base for calibration
final_test_preds_calibrated = ir.predict(stacked_preds)
final_test_preds_calibrated = np.clip(final_test_preds_calibrated, 0, 1)

# Save submission
df_sample[target_col] = final_test_preds_calibrated
df_sample.to_csv("submission.csv", index=False)
print("\nWrote submission.csv (preview):")
print(df_sample.head())

# Save OOF predictions
df_train_p["oof_pred"] = meta_model.predict(meta_X)
df_train_p[["id", "oof_pred"]].to_csv("oof_predictions.csv", index=False)

# Final RMSE
rmse_oof_final = np.sqrt(mean_squared_error(df_train_p[target_col], df_train_p["oof_pred"]))
rmse_prior = np.sqrt(mean_squared_error(df_train_p[target_col], df_train_p["y"]))
print(f"\n--- Final Results ---")
print(f"Baseline prior RMSE: {rmse_prior:.5f}")
print(f"Final Stacked OOF RMSE: {rmse_oof_final:.5f}")
print("\nDone.")
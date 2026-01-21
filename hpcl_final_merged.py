#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 20:01:05 2025

@author: tarak
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare XGB, GPR, DNN on 4 mechanical properties
AND allow manual / file-based prediction + saving outputs.

Authors: Israrul H Hashmi, Tarak Patra
"""

# ========================= IMPORTS =========================
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

import shap
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ========================= BASIC SETTINGS =========================
SEED = 42
DATA_PATH = "/home/tarak/hashmi/hpcl/dataset.xlsx"  # training dataset

# Turn models on/off if needed
RUN_XGB = True
RUN_GPR = True
RUN_DNN = True

# Features and targets (MUST match column names in Excel)
FEATURE_COLS = [
    'PP_H200MA','PP_H350FG','PP_B120MA','PP_C320MN','GF','MAPP','CF'
]

TARGET_COLS = [
    'Tensile Strength at Yield (MPa)',
    'Elongation at yield (%)',
    'Ultimate Tensile Strength (MPa)',
    'Elongation at  break (%)'
]

# Reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ========================= HELPER FUNCTIONS =========================
def parity_plot(y_train, y_pred_train, y_test, y_pred_test,
                target_name, model_name):
    """Create a parity plot (Actual vs Predicted) with consistent style."""
    fig, ax1 = plt.subplots(figsize=(7, 6), dpi=100)

    for side in ["top", "left", "right", "bottom"]:
        ax1.spines[side].set_linewidth(2)

    plt.gcf().subplots_adjust(left=0.25, bottom=0.18)

    ax1.scatter(y_train, y_pred_train, marker='o', color='blue',
                label='Train Data', alpha=0.7, s=200)
    ax1.scatter(y_test, y_pred_test, marker='s', color='red',
                label='Test Data', alpha=0.7, s=200)

    ax1.plot([0, 120], [0, 120], linestyle="dashed", color="green",
             linewidth=2, label='Ideal Fit')

    ax1.set_xlabel(f'Actual {target_name}', fontsize=16)
    ax1.set_ylabel(f'Predicted {target_name}', fontsize=16)
    ax1.set_xlim(0, 120)
    ax1.set_ylim(0, 120)

    ax1.set_xticks(np.arange(20, 120, 20))
    ax1.xaxis.set_major_locator(MultipleLocator(20))
    ax1.yaxis.set_major_locator(MultipleLocator(20))
    ax1.xaxis.set_minor_locator(MultipleLocator(10))
    ax1.yaxis.set_minor_locator(MultipleLocator(10))

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    ax1.tick_params(axis="both", direction="in", length=10, width=4, color="black")
    ax1.tick_params(axis="x", labelsize=25, labelrotation=0)
    ax1.tick_params(axis="y", labelsize=25, labelrotation=0)
    ax1.tick_params(axis='x', which='minor', direction='in', length=5, width=4, color='black')
    ax1.tick_params(axis='y', which='minor', direction='in', length=5, width=4, color='black')

    plt.legend(fontsize=20, frameon=False)
    plt.title(f"{model_name} – {target_name}", fontsize=18)
    plt.show()


def print_summary(model_name, results):
    """Print compact R2 / RMSE summary for a model."""
    print(f"\n\n=========== SUMMARY: {model_name} ===========")
    for res in results:
        print(f"{res['target']}: "
              f"Train R² = {res['r2_train']:.4f}, "
              f"Test R² = {res['r2_test']:.4f}, "
              f"RMSE = {res['rmse']:.4f}")


def predict_all_targets(models, X_scaled, model_type):
    """
    Predict all 4 targets for new data with a list of models.

    models: list of 4 trained models (one per target)
    X_scaled: scaled feature matrix for new data (n_samples x n_features)
    model_type: "xgb", "gpr", or "dnn" (just used to pick predict syntax)
    Returns: array (n_samples x 4)
    """
    n_samples = X_scaled.shape[0]
    n_targets = len(TARGET_COLS)
    preds = np.zeros((n_samples, n_targets))

    for j, model in enumerate(models):
        if model_type == "dnn":
            y_pred = model.predict(X_scaled).ravel()
        else:
            y_pred = model.predict(X_scaled)
        preds[:, j] = y_pred

    return preds


# ========================= LOAD TRAINING DATA =========================
print("Loading training data...")
df = pd.read_excel(DATA_PATH)
X = df[FEATURE_COLS].values
Y = df[TARGET_COLS].values  # shape: (n_samples, 4)

# Shared train/test split
X_train, X_test, Y_train_all, Y_test_all = train_test_split(
    X, Y, test_size=0.2, random_state=SEED
)

# Shared scaler (important to reuse for new-data predictions)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# To store trained models for later prediction
xgb_models = []
gpr_models = []
dnn_models = []

# ========================= 1) XGBOOST =========================
if RUN_XGB:
    xgb_results = []
    for idx, target_name in enumerate(TARGET_COLS):
        print("\n" + "="*70)
        print(f"Training XGBoost for target: {target_name}")
        print("="*70)

        y_train = Y_train_all[:, idx]
        y_test  = Y_test_all[:, idx]

        xg_reg = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.5,
            max_depth=3,
            random_state=SEED
        )

        xg_reg.fit(X_train, y_train)
        xgb_models.append(xg_reg)  # store model

        y_pred_train = xg_reg.predict(X_train)
        y_pred_test  = xg_reg.predict(X_test)

        mse  = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2_tr = r2_score(y_train, y_pred_train)
        r2_te = r2_score(y_test, y_pred_test)

        print(f"Train R²: {r2_tr:.4f}")
        print(f"Test  R²: {r2_te:.4f}")
        print(f"Test  MSE: {mse:.4f}")
        print(f"Test  RMSE: {rmse:.4f}")

        xgb_results.append({
            "target": target_name,
            "r2_train": r2_tr,
            "r2_test": r2_te,
            "rmse": rmse
        })

        parity_plot(y_train, y_pred_train, y_test, y_pred_test,
                    target_name, "XGBoost")

        if idx == 0:  # SHAP only for first target
            print("\nComputing SHAP (XGBoost, first target only)...")
            explainer = shap.TreeExplainer(xg_reg)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test,
                              feature_names=FEATURE_COLS,
                              plot_type="bar")

    print_summary("XGBoost", xgb_results)

# ========================= 2) GPR =========================
if RUN_GPR:
    gpr_results = []
    for idx, target_name in enumerate(TARGET_COLS):
        print("\n" + "="*70)
        print(f"Training GPR for target: {target_name}")
        print("="*70)

        y_train = Y_train_all[:, idx]
        y_test  = Y_test_all[:, idx]

        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0,
                                           length_scale_bounds=(1e-2, 1e2)) \
                 + WhiteKernel()

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=0.0,
            normalize_y=True,
            random_state=SEED
        )

        gpr.fit(X_train, y_train)
        gpr_models.append(gpr)  # store model

        y_pred_train = gpr.predict(X_train)
        y_pred_test  = gpr.predict(X_test)

        mse  = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2_tr = r2_score(y_train, y_pred_train)
        r2_te = r2_score(y_test, y_pred_test)

        print(f"Train R²: {r2_tr:.4f}")
        print(f"Test  R²: {r2_te:.4f}")
        print(f"Test  MSE: {mse:.4f}")
        print(f"Test  RMSE: {rmse:.4f}")

        gpr_results.append({
            "target": target_name,
            "r2_train": r2_tr,
            "r2_test": r2_te,
            "rmse": rmse
        })

        parity_plot(y_train, y_pred_train, y_test, y_pred_test,
                    target_name, "GPR")

        if idx == 0:
            print("\nComputing SHAP (GPR, first target only)...")
            background = X_train if X_train.shape[0] <= 10 else X_train[:10]
            explainer = shap.KernelExplainer(lambda x: gpr.predict(x),
                                             background)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test,
                              feature_names=FEATURE_COLS,
                              plot_type="bar")

    print_summary("GPR", gpr_results)

# ========================= 3) DNN =========================
if RUN_DNN:
    dnn_results = []
    for idx, target_name in enumerate(TARGET_COLS):
        print("\n" + "="*70)
        print(f"Training DNN for target: {target_name}")
        print("="*70)

        y_train = Y_train_all[:, idx:idx+1]
        y_test  = Y_test_all[:, idx:idx+1]

        model = Sequential([
            Dense(28, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(14, activation='relu'),
            Dense(7, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        model.fit(X_train, y_train,
                  epochs=1000,
                  batch_size=3,
                  verbose=0)

        dnn_models.append(model)  # store model

        y_pred_train = model.predict(X_train).ravel()
        y_pred_test  = model.predict(X_test).ravel()

        y_train_flat = y_train.ravel()
        y_test_flat  = y_test.ravel()

        mse  = mean_squared_error(y_test_flat, y_pred_test)
        rmse = np.sqrt(mse)
        r2_tr = r2_score(y_train_flat, y_pred_train)
        r2_te = r2_score(y_test_flat, y_pred_test)

        print(f"Train R²: {r2_tr:.4f}")
        print(f"Test  R²: {r2_te:.4f}")
        print(f"Test  MSE: {mse:.4f}")
        print(f"Test  RMSE: {rmse:.4f}")

        dnn_results.append({
            "target": target_name,
            "r2_train": r2_tr,
            "r2_test": r2_te,
            "rmse": rmse
        })

        parity_plot(y_train_flat, y_pred_train, y_test_flat, y_pred_test,
                    target_name, "DNN")

        if idx == 0:
            print("\nComputing SHAP (DNN, first target only)...")
            background = X_train if X_train.shape[0] <= 10 else X_train[:10]
            explainer = shap.KernelExplainer(
                lambda x: model.predict(x).ravel(),
                background
            )
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test,
                              feature_names=FEATURE_COLS,
                              plot_type="bar")

    print_summary("DNN", dnn_results)

# ==========================================================
#  MANUAL / FILE-BASED PREDICTION AND SAVE OUTPUTS
# ==========================================================
"""
Here we use the trained models to predict outputs for NEW compositions.

Two options:
1) Manual input (single row defined in code)
2) Read many rows from 'datafile.xlsx'

Each row must sum to 1 (e.g., mass or volume fractions).
Predictions are saved to:
- output_XGB.txt
- output_GPR.txt
- output_DNN.txt
"""

USE_MANUAL_INPUT = True   # Set False to use datafile.xlsx

if USE_MANUAL_INPUT:
    # ---- Manual input example (one composition) ----
    # Order of values MUST follow FEATURE_COLS
    # Example: [PP_H200MA, PP_H350FG, PP_B120MA, PP_C320MN, GF, MAPP, CF]
    manual_feature = np.array([0.58, 0.0, 0.0, 0.0, 0.40, 0.02, 0.0])
    new_X_raw = manual_feature.reshape(1, -1)  # shape (1, 7)
else:
    # ---- Read new data from file ----
    new_df = pd.read_excel("datafile.xlsx")
    new_X_raw = new_df[FEATURE_COLS].values  # shape (n_new, 7)

# ---- Check each row sums to 1 (like your old code) ----
row_sums = new_X_raw.sum(axis=1)
for i, s in enumerate(row_sums):
    if not np.isclose(s, 1.0, atol=1e-6):
        raise ValueError(
            f"New data row {i} does not sum to 1 (sum = {s}). "
            "All rows must sum to 1."
        )

print("All new-data rows sum to 1. Continuing with prediction...")

# ---- Scale new data using the SAME scaler as training ----
new_X_scaled = scaler.transform(new_X_raw)

# ---- Predict and save outputs for each model type ----
header_str = ", ".join(TARGET_COLS)

if RUN_XGB and xgb_models:
    preds_xgb = predict_all_targets(xgb_models, new_X_scaled, model_type="xgb")
    np.savetxt("output_XGB.txt", preds_xgb, header=header_str, fmt="%.5f")
    print("Saved XGB predictions to output_XGB.txt")

if RUN_GPR and gpr_models:
    preds_gpr = predict_all_targets(gpr_models, new_X_scaled, model_type="gpr")
    np.savetxt("output_GPR.txt", preds_gpr, header=header_str, fmt="%.5f")
    print("Saved GPR predictions to output_GPR.txt")

if RUN_DNN and dnn_models:
    preds_dnn = predict_all_targets(dnn_models, new_X_scaled, model_type="dnn")
    np.savetxt("output_DNN.txt", preds_dnn, header=header_str, fmt="%.5f")
    print("Saved DNN predictions to output_DNN.txt")

print("\nDone.")

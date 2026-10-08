"""
model_MLP.py
------------
Entrena y evalúa modelos MLP (Multi-Layer Perceptron) para clasificación
de atomtype y regresión de charge, comparándolos con Random Forest.

Diferencias clave respecto al modelo RF:
  - Requiere normalización de features (StandardScaler)
  - Más sensible a hiperparámetros
  - Mismo split 80/20 y CV 5-fold para comparación justa

Salidas:
  results_MLP_{label}/
    ├── clf_MLP_{label}.joblib
    ├── reg_MLP_{label}.joblib
    ├── scaler_clf_{label}.joblib     escalador para clasificación
    ├── scaler_reg_{label}.joblib     escalador para regresión
    ├── metrics_MLP_{label}.json
    ├── fig_confusion_matrix.png
    ├── fig_feature_importance_clf.png  (permutation importance)
    ├── fig_feature_importance_reg.png
    ├── fig_charge_predicted_vs_real.png
    ├── fig_charge_residuals.png
    ├── fig_crossval_scores.png
    ├── fig_atomtype_distribution.png
    └── fig_comparison_RF_vs_MLP.png   comparación directa

Uso:
    python model_MLP.py --input processed/aa_clean.csv
                        --output_dir results_MLP_AA/
                        --label AA
                        --rf_metrics results_AA/metrics_AA.json

Dependencias:
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate, train_test_split,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"]  = 11

# =============================================================================
# CONFIG
# =============================================================================

TARGET_CLF  = "atomtype"
TARGET_REG  = "charge"
TEST_SIZE   = 0.20
RANDOM_SEED = 42
CV_FOLDS    = 5
MIN_SAMPLES = 2
TOP_N_FEATURES = 20

# Hiperparámetros MLP
# Arquitectura: dos capas ocultas de 128 y 64 neuronas
# relu: función de activación estándar para capas ocultas
# adam: optimizador adaptativo, robusto con datasets pequeños
# max_iter=1000: suficientes iteraciones para convergencia
# early_stopping: detiene el entrenamiento si la validación no mejora
#   evita overfitting sin necesidad de tunear manualmente las epochs
# alpha: regularización L2, importante con datasets pequeños
MLP_CLF_PARAMS = dict(
    hidden_layer_sizes  = (128, 64),
    activation          = "relu",
    solver              = "adam",
    alpha               = 0.001,
    max_iter            = 1000,
    early_stopping      = False,    # ← cambiar a False
    random_state        = RANDOM_SEED,
)

MLP_REG_PARAMS = dict(
    hidden_layer_sizes  = (128, 64),
    activation          = "relu",
    solver              = "adam",
    alpha               = 0.001,
    max_iter            = 1000,
    early_stopping      = False,    # ← cambiar a False
    random_state        = RANDOM_SEED,
)


# =============================================================================
# HELPERS
# =============================================================================

def save(fig, out_dir, name):
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def section(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")


# =============================================================================
# CARGA Y SPLIT
# =============================================================================

def load_and_split(csv_path: Path):
    df = pd.read_csv(csv_path)
    print(f"Dataset cargado: {df.shape}")

    # Eliminar clases con 1 instancia
    counts  = df[TARGET_CLF].value_counts()
    removed = sorted(counts[counts < MIN_SAMPLES].index.tolist())
    mask    = df[TARGET_CLF].isin(counts[counts >= MIN_SAMPLES].index)
    df      = df[mask].reset_index(drop=True)
    if removed:
        print(f"  Clases eliminadas (1 instancia): {removed}")
    print(f"  Átomos restantes: {len(df)}")

    feature_cols = [c for c in df.columns if c not in [TARGET_CLF, TARGET_REG]]
    X     = df[feature_cols]
    y_clf = df[TARGET_CLF]
    y_reg = df[TARGET_REG]

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = \
        train_test_split(
            X, y_clf, y_reg,
            test_size    = TEST_SIZE,
            random_state = RANDOM_SEED,
            stratify     = y_clf,
        )

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    return X, X_train, X_test, y_clf, y_clf_train, y_clf_test, \
           y_reg, y_reg_train, y_reg_test, feature_cols


# =============================================================================
# NORMALIZACIÓN
# =============================================================================

def normalize(X_train, X_test, X_full):
    """
    StandardScaler: escala cada feature a media=0 y std=1.
    Obligatorio para MLP — las redes neuronales son sensibles a la escala
    de las features. Sin normalización, features con valores grandes
    (como mass ~12-35) dominarían sobre features pequeñas (como is_planar=0/1).
    El scaler se ajusta SOLO sobre train para evitar data leakage.
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    X_full_sc  = scaler.transform(X_full)
    return X_train_sc, X_test_sc, X_full_sc, scaler


# =============================================================================
# CLASIFICACIÓN
# =============================================================================

def train_and_evaluate_classifier(X_train_sc, X_test_sc, X_full_sc,
                                   y_clf_train, y_clf_test, y_clf,
                                   feature_cols, out_dir, label):
    section("Entrenando clasificador MLP (atomtype)")
    clf = MLPClassifier(**MLP_CLF_PARAMS)
    clf.fit(X_train_sc, y_clf_train)
    print(f"  ✓ Convergió en {clf.n_iter_} iteraciones")

    y_pred_train = clf.predict(X_train_sc)
    y_pred_test  = clf.predict(X_test_sc)

    acc_train = accuracy_score(y_clf_train, y_pred_train)
    acc_test  = accuracy_score(y_clf_test,  y_pred_test)
    f1_macro  = f1_score(y_clf_test, y_pred_test, average="macro",    zero_division=0)
    f1_w      = f1_score(y_clf_test, y_pred_test, average="weighted", zero_division=0)

    section("Evaluación — Clasificación MLP")
    print(f"  Accuracy train:  {acc_train:.4f}")
    print(f"  Accuracy test:   {acc_test:.4f}")
    print(f"  F1 macro:        {f1_macro:.4f}")
    print(f"  F1 weighted:     {f1_w:.4f}")
    print()
    print(classification_report(y_clf_test, y_pred_test, zero_division=0))

    # Cross-validation
    section("Cross-validation — Clasificación MLP")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_results = cross_validate(
        MLPClassifier(**MLP_CLF_PARAMS), X_full_sc, y_clf,
        cv=skf, scoring=["accuracy", "f1_macro", "f1_weighted"],
        return_train_score=True, n_jobs=-1,
    )
    cv_acc = cv_results["test_accuracy"]
    cv_f1m = cv_results["test_f1_macro"]
    cv_f1w = cv_results["test_f1_weighted"]
    print(f"  CV Accuracy:    {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"  CV F1 macro:    {cv_f1m.mean():.4f} ± {cv_f1m.std():.4f}")
    print(f"  CV F1 weighted: {cv_f1w.mean():.4f} ± {cv_f1w.std():.4f}")

    # Matriz de confusión
    classes = sorted(y_clf.unique())
    cm      = confusion_matrix(y_clf_test, y_pred_test, labels=classes)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — MLP {label} (test set)")
    plt.tight_layout()
    save(fig, out_dir, "fig_confusion_matrix.png")

    # Permutation importance (MLP no tiene feature_importances_ nativo)
    section("Permutation importance — Clasificación MLP")
    print("  Calculando (puede tardar unos segundos)...")
    perm = permutation_importance(clf, X_test_sc, y_clf_test,
                                  n_repeats=10, random_state=RANDOM_SEED,
                                  scoring="accuracy")
    imp = pd.Series(perm.importances_mean, index=feature_cols).nlargest(TOP_N_FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp.index, imp.values, color=sns.color_palette("muted", len(imp)))
    ax.set_xlabel("Importancia (Permutation — Accuracy drop)")
    ax.set_title(f"Feature importance — Clasificación MLP {label}\n(top {TOP_N_FEATURES})")
    plt.tight_layout()
    save(fig, out_dir, "fig_feature_importance_clf.png")

    # CV scores por fold
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    folds = np.arange(1, CV_FOLDS + 1)
    for ax, scores, title, color in zip(
        axes, [cv_acc, cv_f1m, cv_f1w],
        ["Accuracy", "F1 macro", "F1 weighted"],
        ["#378ADD", "#1D9E75", "#534AB7"],
    ):
        ax.bar(folds, scores, color=color, alpha=0.8)
        ax.axhline(scores.mean(), color="red", linestyle="--",
                   label=f"Media: {scores.mean():.3f}")
        ax.set_xticks(folds)
        ax.set_xlabel("Fold")
        ax.set_ylabel(title)
        ax.set_title(f"{title} por fold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
    plt.suptitle(f"Cross-validation ({CV_FOLDS} folds) — Clasificación MLP {label}", y=1.02)
    plt.tight_layout()
    save(fig, out_dir, "fig_crossval_scores.png")

    return {
        "accuracy_train":      round(acc_train, 4),
        "accuracy_test":       round(acc_test,  4),
        "f1_macro":            round(f1_macro,  4),
        "f1_weighted":         round(f1_w,      4),
        "cv_accuracy_mean":    round(float(cv_acc.mean()), 4),
        "cv_accuracy_std":     round(float(cv_acc.std()),  4),
        "cv_f1_macro_mean":    round(float(cv_f1m.mean()), 4),
        "cv_f1_macro_std":     round(float(cv_f1m.std()),  4),
        "cv_f1_weighted_mean": round(float(cv_f1w.mean()), 4),
        "cv_f1_weighted_std":  round(float(cv_f1w.std()),  4),
        "n_iterations":        clf.n_iter_,
        "classification_report": classification_report(
            y_clf_test, y_pred_test, zero_division=0, output_dict=True
        ),
    }, clf


# =============================================================================
# REGRESIÓN
# =============================================================================

def train_and_evaluate_regressor(X_train_sc, X_test_sc, X_full_sc,
                                  y_reg_train, y_reg_test, y_reg,
                                  feature_cols, out_dir, label):
    section("Entrenando regresor MLP (charge)")
    reg = MLPRegressor(**MLP_REG_PARAMS)
    reg.fit(X_train_sc, y_reg_train)
    print(f"  ✓ Convergió en {reg.n_iter_} iteraciones")

    y_pred_train = reg.predict(X_train_sc)
    y_pred_test  = reg.predict(X_test_sc)

    mae_train  = mean_absolute_error(y_reg_train, y_pred_train)
    mae_test   = mean_absolute_error(y_reg_test,  y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_reg_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_reg_test,  y_pred_test))
    r2_train   = r2_score(y_reg_train, y_pred_train)
    r2_test    = r2_score(y_reg_test,  y_pred_test)

    section("Evaluación — Regresión MLP")
    print(f"  MAE  train: {mae_train:.4f} e  |  test: {mae_test:.4f} e")
    print(f"  RMSE train: {rmse_train:.4f} e  |  test: {rmse_test:.4f} e")
    print(f"  R²   train: {r2_train:.4f}    |  test: {r2_test:.4f}")

    # Cross-validation
    section("Cross-validation — Regresión MLP")
    cv_mae  = -cross_val_score(MLPRegressor(**MLP_REG_PARAMS), X_full_sc, y_reg,
                               cv=CV_FOLDS, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_rmse = np.sqrt(-cross_val_score(MLPRegressor(**MLP_REG_PARAMS), X_full_sc, y_reg,
                                       cv=CV_FOLDS, scoring="neg_mean_squared_error", n_jobs=-1))
    cv_r2   = cross_val_score(MLPRegressor(**MLP_REG_PARAMS), X_full_sc, y_reg,
                              cv=CV_FOLDS, scoring="r2", n_jobs=-1)
    print(f"  CV MAE:  {cv_mae.mean():.4f} ± {cv_mae.std():.4f} e")
    print(f"  CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f} e")
    print(f"  CV R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    # Scatter real vs predicho
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_reg_test, y_pred_test, alpha=0.6, color="#378ADD",
               edgecolors="white", linewidth=0.3, s=50)
    lims = [min(y_reg_test.min(), y_pred_test.min()) - 0.05,
            max(y_reg_test.max(), y_pred_test.max()) + 0.05]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Predicción perfecta")
    ax.set_xlabel("Charge real (e)")
    ax.set_ylabel("Charge predicha (e)")
    ax.set_title(f"Charge real vs predicha — MLP {label}\n"
                 f"MAE={mae_test:.4f} | RMSE={rmse_test:.4f} | R²={r2_test:.4f}")
    ax.legend()
    plt.tight_layout()
    save(fig, out_dir, "fig_charge_predicted_vs_real.png")

    # Residuos
    residuals = y_reg_test - y_pred_test
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred_test, residuals, alpha=0.6, color="#534AB7",
                    edgecolors="white", linewidth=0.3, s=50)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Charge predicha (e)")
    axes[0].set_ylabel("Residuo (real - predicho)")
    axes[0].set_title("Residuos vs Charge predicha")
    axes[1].hist(residuals, bins=25, color="#534AB7", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].axvline(residuals.mean(), color="orange", linestyle="--",
                    linewidth=1, label=f"Media: {residuals.mean():.4f}")
    axes[1].set_xlabel("Residuo (e)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de residuos")
    axes[1].legend()
    plt.suptitle(f"Análisis de residuos — MLP {label}", y=1.02)
    plt.tight_layout()
    save(fig, out_dir, "fig_charge_residuals.png")

    # Permutation importance regresión
    section("Permutation importance — Regresión MLP")
    print("  Calculando...")
    perm = permutation_importance(reg, X_test_sc, y_reg_test,
                                  n_repeats=10, random_state=RANDOM_SEED,
                                  scoring="r2")
    imp = pd.Series(perm.importances_mean, index=feature_cols).nlargest(TOP_N_FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp.index, imp.values, color=sns.color_palette("muted", len(imp)))
    ax.set_xlabel("Importancia (Permutation — R² drop)")
    ax.set_title(f"Feature importance — Regresión MLP {label}\n(top {TOP_N_FEATURES})")
    plt.tight_layout()
    save(fig, out_dir, "fig_feature_importance_reg.png")

    return {
        "mae_train":    round(mae_train,  4),
        "mae_test":     round(mae_test,   4),
        "rmse_train":   round(rmse_train, 4),
        "rmse_test":    round(rmse_test,  4),
        "r2_train":     round(r2_train,   4),
        "r2_test":      round(r2_test,    4),
        "cv_mae_mean":  round(float(cv_mae.mean()),  4),
        "cv_mae_std":   round(float(cv_mae.std()),   4),
        "cv_rmse_mean": round(float(cv_rmse.mean()), 4),
        "cv_rmse_std":  round(float(cv_rmse.std()),  4),
        "cv_r2_mean":   round(float(cv_r2.mean()),   4),
        "cv_r2_std":    round(float(cv_r2.std()),    4),
        "n_iterations": reg.n_iter_,
    }, reg


# =============================================================================
# FIGURA COMPARACIÓN RF vs MLP
# =============================================================================

def plot_comparison(metrics_mlp, rf_metrics_path, label, out_dir):
    """
    Genera figura comparativa RF vs MLP para las métricas clave.
    """
    if rf_metrics_path is None or not Path(rf_metrics_path).exists():
        print("  [SKIP] No se encontró metrics_RF — omitiendo figura comparativa.")
        return

    with open(rf_metrics_path) as f:
        metrics_rf = json.load(f)

    section("Figura comparativa RF vs MLP")

    # Métricas a comparar
    clf_metrics = {
        "Accuracy test":   [metrics_rf["classification"]["accuracy_test"],
                            metrics_mlp["classification"]["accuracy_test"]],
        "F1 macro test":   [metrics_rf["classification"]["f1_macro"],
                            metrics_mlp["classification"]["f1_macro"]],
        "CV Accuracy":     [metrics_rf["classification"]["cv_accuracy_mean"],
                            metrics_mlp["classification"]["cv_accuracy_mean"]],
    }
    reg_metrics = {
        "MAE test (e)":    [metrics_rf["regression"]["mae_test"],
                            metrics_mlp["regression"]["mae_test"]],
        "RMSE test (e)":   [metrics_rf["regression"]["rmse_test"],
                            metrics_mlp["regression"]["rmse_test"]],
        "R² test":         [metrics_rf["regression"]["r2_test"],
                            metrics_mlp["regression"]["r2_test"]],
        "CV R²":           [metrics_rf["regression"]["cv_r2_mean"],
                            metrics_mlp["regression"]["cv_r2_mean"]],
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#378ADD", "#D85A30"]
    x_clf = np.arange(len(clf_metrics))
    x_reg = np.arange(len(reg_metrics))
    w = 0.35

    # Clasificación
    ax = axes[0]
    rf_vals  = [v[0] for v in clf_metrics.values()]
    mlp_vals = [v[1] for v in clf_metrics.values()]
    ax.bar(x_clf - w/2, rf_vals,  w, label="RF",  color=colors[0])
    ax.bar(x_clf + w/2, mlp_vals, w, label="MLP", color=colors[1])
    ax.set_xticks(x_clf)
    ax.set_xticklabels(list(clf_metrics.keys()), rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Clasificación — RF vs MLP\n{label}")
    ax.set_ylim(0, 1.1)
    ax.legend()
    for i, (r, m) in enumerate(zip(rf_vals, mlp_vals)):
        ax.text(i - w/2, r + 0.01, f"{r:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, m + 0.01, f"{m:.3f}", ha="center", fontsize=8)

    # Regresión
    ax = axes[1]
    rf_vals  = [v[0] for v in reg_metrics.values()]
    mlp_vals = [v[1] for v in reg_metrics.values()]
    ax.bar(x_reg - w/2, rf_vals,  w, label="RF",  color=colors[0])
    ax.bar(x_reg + w/2, mlp_vals, w, label="MLP", color=colors[1])
    ax.set_xticks(x_reg)
    ax.set_xticklabels(list(reg_metrics.keys()), rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Regresión — RF vs MLP\n{label}")
    ax.legend()
    for i, (r, m) in enumerate(zip(rf_vals, mlp_vals)):
        ax.text(i - w/2, r + 0.005, f"{r:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, m + 0.005, f"{m:.3f}", ha="center", fontsize=8)

    plt.suptitle(f"Comparación RF vs MLP — {label}", y=1.02)
    plt.tight_layout()
    save(fig, out_dir, "fig_comparison_RF_vs_MLP.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Modelo MLP con comparación vs Random Forest"
    )
    parser.add_argument("--input",      required=True,
                        help="CSV limpio (aa_clean.csv o ua_clean.csv)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label",      required=True,
                        help="AA o UA")
    parser.add_argument("--rf_metrics", default=None,
                        help="Path a metrics_AA.json o metrics_UA.json (opcional)")
    args    = parser.parse_args()
    label   = args.label
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  MODELO MLP — {label}")
    print(f"{'='*55}")

    # Cargar y dividir
    (X, X_train, X_test,
     y_clf, y_clf_train, y_clf_test,
     y_reg, y_reg_train, y_reg_test,
     feature_cols) = load_and_split(Path(args.input))

    # Normalizar — dos scalers separados para clf y reg
    X_train_sc_clf, X_test_sc_clf, X_full_sc_clf, scaler_clf = \
        normalize(X_train, X_test, X)
    X_train_sc_reg, X_test_sc_reg, X_full_sc_reg, scaler_reg = \
        normalize(X_train, X_test, X)

    # Distribución de clases
    counts_full  = y_clf.value_counts()
    counts_train = y_clf_train.value_counts().reindex(counts_full.index, fill_value=0)
    counts_test  = y_clf_test.value_counts().reindex(counts_full.index,  fill_value=0)
    x = np.arange(len(counts_full))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, counts_train.values, w, label="Train", color="#378ADD")
    ax.bar(x + w/2, counts_test.values,  w, label="Test",  color="#D85A30")
    ax.set_xticks(x)
    ax.set_xticklabels(counts_full.index, rotation=45, ha="right")
    ax.set_title(f"Distribución de clases — MLP {label} (Train vs Test)")
    ax.legend()
    plt.tight_layout()
    save(fig, out_dir, "fig_atomtype_distribution.png")

    # Clasificación
    metrics_clf, clf = train_and_evaluate_classifier(
        X_train_sc_clf, X_test_sc_clf, X_full_sc_clf,
        y_clf_train, y_clf_test, y_clf,
        feature_cols, out_dir, label
    )

    # Regresión
    metrics_reg, reg = train_and_evaluate_regressor(
        X_train_sc_reg, X_test_sc_reg, X_full_sc_reg,
        y_reg_train, y_reg_test, y_reg,
        feature_cols, out_dir, label
    )

    # Guardar modelos y scalers
    section("Guardando modelos")
    dump(clf,        out_dir / f"clf_MLP_{label}.joblib")
    dump(reg,        out_dir / f"reg_MLP_{label}.joblib")
    dump(scaler_clf, out_dir / f"scaler_clf_{label}.joblib")
    dump(scaler_reg, out_dir / f"scaler_reg_{label}.joblib")
    print(f"  → clf_MLP_{label}.joblib")
    print(f"  → reg_MLP_{label}.joblib")
    print(f"  → scaler_clf_{label}.joblib")
    print(f"  → scaler_reg_{label}.joblib")

    # Guardar métricas
    metrics = {
        "dataset":       label,
        "model":         "MLP",
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "n_features":    len(feature_cols),
        "mlp_clf_params": MLP_CLF_PARAMS,
        "mlp_reg_params": MLP_REG_PARAMS,
        "classification": metrics_clf,
        "regression":     metrics_reg,
    }
    metrics_path = out_dir / f"metrics_MLP_{label}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → metrics_MLP_{label}.json")

    # Figura comparativa
    plot_comparison(metrics, args.rf_metrics, label, out_dir)

    # Resumen final
    section("RESUMEN FINAL — MLP")
    print(f"  Clasificación (atomtype):")
    print(f"    Accuracy test:   {metrics_clf['accuracy_test']:.4f}")
    print(f"    F1 macro test:   {metrics_clf['f1_macro']:.4f}")
    print(f"    CV Accuracy:     {metrics_clf['cv_accuracy_mean']:.4f} "
          f"± {metrics_clf['cv_accuracy_std']:.4f}")
    print(f"\n  Regresión (charge):")
    print(f"    MAE test:        {metrics_reg['mae_test']:.4f} e")
    print(f"    RMSE test:       {metrics_reg['rmse_test']:.4f} e")
    print(f"    R² test:         {metrics_reg['r2_test']:.4f}")
    print(f"    CV R²:           {metrics_reg['cv_r2_mean']:.4f} "
          f"± {metrics_reg['cv_r2_std']:.4f}")

    print(f"\n✔ Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()
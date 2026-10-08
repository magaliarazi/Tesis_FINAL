"""
model_AA.py
-----------
Entrena y evalúa modelos Random Forest para el dataset All-Atom (AA).

Tareas:
  - Clasificación: predicción de atomtype
  - Regresión:     predicción de charge

Estrategia de evaluación:
  - Split 80/20 estratificado por atomtype → métricas finales en test
  - 5-fold cross-validation → métricas robustas para la tesis

Salidas:
  results_AA/
    ├── clf_AA.joblib                 modelo de clasificación
    ├── reg_AA.joblib                 modelo de regresión
    ├── metrics_AA.json               métricas completas
    ├── fig_atomtype_distribution.png distribución de clases
    ├── fig_confusion_matrix.png      matriz de confusión
    ├── fig_feature_importance_clf.png feature importance clasificación
    ├── fig_feature_importance_reg.png feature importance regresión
    ├── fig_charge_predicted_vs_real.png scatter real vs predicho
    ├── fig_charge_residuals.png      residuos de regresión
    └── fig_crossval_scores.png       scores de cross-validation

Uso:
    python model_AA.py --input processed/aa_clean.csv --output_dir results_AA/

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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate, train_test_split,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"]  = 11

LABEL = "AA"

# =============================================================================
# CONFIG
# =============================================================================

TARGET_CLF = "atomtype"
TARGET_REG = "charge"
TEST_SIZE   = 0.20
RANDOM_SEED = 42
CV_FOLDS    = 5

# Hiperparámetros Random Forest
# n_estimators=150: suficientes árboles para estabilizar las predicciones
# max_depth=10: árboles completos, dejar que el modelo encuentre profundidad óptima
# min_samples_leaf=1: default, permite aprender de clases raras
# class_weight="balanced": pondera inversamente la frecuencia de cada clase
#   para compensar el desbalance severo (ratio 157x en AA)
RF_CLF_PARAMS = dict(
    n_estimators  = 150,
    max_depth     = 10,
    min_samples_leaf = 1,
    class_weight  = "balanced",
    random_state  = RANDOM_SEED,
    n_jobs        = -1,
)

RF_REG_PARAMS = dict(
    n_estimators  = 150,
    max_depth     = 10,
    min_samples_leaf = 1,
    random_state  = RANDOM_SEED,
    n_jobs        = -1,
)

TOP_N_FEATURES = 20   # cuántas features mostrar en los gráficos de importancia


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
    """
    Carga el dataset limpio y separa features, targets y conjuntos train/test.

    El split es estratificado por atomtype para garantizar que cada clase
    esté representada proporcionalmente en train y test, lo cual es crítico
    con clases de baja frecuencia.
    """
    df = pd.read_csv(csv_path)
    print(f"Dataset cargado: {df.shape}")
    print(f"  Clases atomtype: {df[TARGET_CLF].nunique()}")
    print(f"  Rango charge:    [{df[TARGET_REG].min():.3f}, {df[TARGET_REG].max():.3f}]")

    feature_cols = [c for c in df.columns if c not in [TARGET_CLF, TARGET_REG]]
    X = df[feature_cols]
    y_clf = df[TARGET_CLF]
    y_reg = df[TARGET_REG]

    # Eliminar clases con menos de 2 instancias (imposibles de estratificar)
    MIN_SAMPLES = 2
    counts  = y_clf.value_counts()
    removed = sorted(counts[counts < MIN_SAMPLES].index.tolist())
    mask    = y_clf.isin(counts[counts >= MIN_SAMPLES].index)
    X       = X[mask].reset_index(drop=True)
    y_clf   = y_clf[mask].reset_index(drop=True)
    y_reg   = y_reg[mask].reset_index(drop=True)
    print(f"  Clases eliminadas (1 instancia): {removed}")
    print(f"  Átomos eliminados: {(~mask).sum()}")
    print(f"  Átomos restantes:  {len(X)}")

    # Debug — identificar columnas no numéricas
    non_numeric = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"  Columnas no numéricas: {non_numeric}")

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = \
        train_test_split(
            X, y_clf, y_reg,
            test_size    = TEST_SIZE,
            random_state = RANDOM_SEED,
            stratify     = y_clf,
        )

    print(f"\n  Train: {len(X_train)} átomos ({len(X_train)/len(X)*100:.0f}%)")
    print(f"  Test:  {len(X_test)} átomos ({len(X_test)/len(X)*100:.0f}%)")

    return X, X_train, X_test, y_clf, y_clf_train, y_clf_test, \
           y_reg, y_reg_train, y_reg_test, feature_cols


# =============================================================================
# VISUALIZACIÓN 1 — distribución de clases
# =============================================================================

def plot_class_distribution(y_clf, y_clf_train, y_clf_test, out_dir):
    section("Distribución de clases")

    counts_full  = y_clf.value_counts()
    counts_train = y_clf_train.value_counts()
    counts_test  = y_clf_test.value_counts()

    # Alinear índices
    all_classes = counts_full.index
    counts_train = counts_train.reindex(all_classes, fill_value=0)
    counts_test  = counts_test.reindex(all_classes, fill_value=0)

    x = np.arange(len(all_classes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, counts_train.values, w, label="Train", color="#378ADD")
    ax.bar(x + w/2, counts_test.values,  w, label="Test",  color="#D85A30")
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha="right")
    ax.set_xlabel("atomtype")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución de clases — {LABEL} (Train vs Test)")
    ax.legend()
    plt.tight_layout()
    save(fig, out_dir, "fig_atomtype_distribution.png")


# =============================================================================
# CLASIFICACIÓN
# =============================================================================

def train_classifier(X_train, y_train):
    section("Entrenando clasificador (atomtype)")
    clf = RandomForestClassifier(**RF_CLF_PARAMS)
    clf.fit(X_train, y_train)
    print("  ✓ Clasificador entrenado")
    return clf


def evaluate_classifier(clf, X_train, X_test, y_train, y_test,
                         X_full, y_full, feature_cols, out_dir):
    section("Evaluación — Clasificación")

    y_pred_train = clf.predict(X_train)
    y_pred_test  = clf.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test  = accuracy_score(y_test,  y_pred_test)
    f1_macro  = f1_score(y_test, y_pred_test, average="macro",    zero_division=0)
    f1_w      = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)

    print(f"  Accuracy train:  {acc_train:.4f}")
    print(f"  Accuracy test:   {acc_test:.4f}")
    print(f"  F1 macro:        {f1_macro:.4f}")
    print(f"  F1 weighted:     {f1_w:.4f}")
    print()
    print(classification_report(y_test, y_pred_test, zero_division=0))

    # Cross-validation
    section("Cross-validation — Clasificación")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_results = cross_validate(
        RandomForestClassifier(**RF_CLF_PARAMS), X_full, y_full,
        cv=skf, scoring=["accuracy", "f1_macro", "f1_weighted"],
        return_train_score=True, n_jobs=-1,
    )
    cv_acc   = cv_results["test_accuracy"]
    cv_f1m   = cv_results["test_f1_macro"]
    cv_f1w   = cv_results["test_f1_weighted"]
    print(f"  CV Accuracy:    {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"  CV F1 macro:    {cv_f1m.mean():.4f} ± {cv_f1m.std():.4f}")
    print(f"  CV F1 weighted: {cv_f1w.mean():.4f} ± {cv_f1w.std():.4f}")

    # Figura — Matriz de confusión
    classes   = sorted(y_full.unique())
    cm        = confusion_matrix(y_test, y_pred_test, labels=classes)
    cm_norm   = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {LABEL} (test set)\n"
                 f"(valores: conteo, color: proporción por fila)")
    plt.tight_layout()
    save(fig, out_dir, "fig_confusion_matrix.png")

    # Figura — Feature importance clasificación
    plot_feature_importance(clf, feature_cols, out_dir,
                            "fig_feature_importance_clf.png",
                            f"Feature importance — Clasificación {LABEL}")

    # Figura — Cross-validation scores
    plot_crossval(cv_acc, cv_f1m, cv_f1w, out_dir)

    return {
        "accuracy_train":  round(acc_train, 4),
        "accuracy_test":   round(acc_test,  4),
        "f1_macro":        round(f1_macro,  4),
        "f1_weighted":     round(f1_w,      4),
        "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
        "cv_accuracy_std":  round(float(cv_acc.std()),  4),
        "cv_f1_macro_mean": round(float(cv_f1m.mean()), 4),
        "cv_f1_macro_std":  round(float(cv_f1m.std()),  4),
        "cv_f1_weighted_mean": round(float(cv_f1w.mean()), 4),
        "cv_f1_weighted_std":  round(float(cv_f1w.std()),  4),
        "classification_report": classification_report(
            y_test, y_pred_test, zero_division=0, output_dict=True
        ),
    }


# =============================================================================
# REGRESIÓN
# =============================================================================

def train_regressor(X_train, y_train):
    section("Entrenando regresor (charge)")
    reg = RandomForestRegressor(**RF_REG_PARAMS)
    reg.fit(X_train, y_train)
    print("  ✓ Regresor entrenado")
    return reg


def evaluate_regressor(reg, X_train, X_test, y_train, y_test,
                        X_full, y_full, feature_cols, out_dir):
    section("Evaluación — Regresión")

    y_pred_train = reg.predict(X_train)
    y_pred_test  = reg.predict(X_test)

    mae_train  = mean_absolute_error(y_train, y_pred_train)
    mae_test   = mean_absolute_error(y_test,  y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    r2_train   = r2_score(y_train, y_pred_train)
    r2_test    = r2_score(y_test,  y_pred_test)

    print(f"  MAE  train: {mae_train:.4f} e  |  test: {mae_test:.4f} e")
    print(f"  RMSE train: {rmse_train:.4f} e  |  test: {rmse_test:.4f} e")
    print(f"  R²   train: {r2_train:.4f}    |  test: {r2_test:.4f}")

    # Cross-validation
    section("Cross-validation — Regresión")
    cv_mae  = -cross_val_score(
        RandomForestRegressor(**RF_REG_PARAMS), X_full, y_full,
        cv=CV_FOLDS, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    cv_rmse = np.sqrt(-cross_val_score(
        RandomForestRegressor(**RF_REG_PARAMS), X_full, y_full,
        cv=CV_FOLDS, scoring="neg_mean_squared_error", n_jobs=-1
    ))
    cv_r2 = cross_val_score(
        RandomForestRegressor(**RF_REG_PARAMS), X_full, y_full,
        cv=CV_FOLDS, scoring="r2", n_jobs=-1
    )
    print(f"  CV MAE:  {cv_mae.mean():.4f} ± {cv_mae.std():.4f} e")
    print(f"  CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f} e")
    print(f"  CV R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    # Figura — Predicho vs Real
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred_test, alpha=0.6, color="#378ADD",
               edgecolors="white", linewidth=0.3, s=50, label="Test")
    lims = [min(y_test.min(), y_pred_test.min()) - 0.05,
            max(y_test.max(), y_pred_test.max()) + 0.05]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Predicción perfecta")
    ax.set_xlabel("Charge real (e)")
    ax.set_ylabel("Charge predicha (e)")
    ax.set_title(f"Charge real vs predicha — {LABEL}\n"
                 f"MAE={mae_test:.4f} | RMSE={rmse_test:.4f} | R²={r2_test:.4f}")
    ax.legend()
    plt.tight_layout()
    save(fig, out_dir, "fig_charge_predicted_vs_real.png")

    # Figura — Residuos
    residuals = y_test - y_pred_test
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_pred_test, residuals, alpha=0.6,
                    color="#534AB7", edgecolors="white", linewidth=0.3, s=50)
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

    plt.suptitle(f"Análisis de residuos — {LABEL}", y=1.02)
    plt.tight_layout()
    save(fig, out_dir, "fig_charge_residuals.png")

    # Figura — Feature importance regresión
    plot_feature_importance(reg, feature_cols, out_dir,
                            "fig_feature_importance_reg.png",
                            f"Feature importance — Regresión {LABEL}")

    return {
        "mae_train":   round(mae_train,  4),
        "mae_test":    round(mae_test,   4),
        "rmse_train":  round(rmse_train, 4),
        "rmse_test":   round(rmse_test,  4),
        "r2_train":    round(r2_train,   4),
        "r2_test":     round(r2_test,    4),
        "cv_mae_mean":  round(float(cv_mae.mean()),  4),
        "cv_mae_std":   round(float(cv_mae.std()),   4),
        "cv_rmse_mean": round(float(cv_rmse.mean()), 4),
        "cv_rmse_std":  round(float(cv_rmse.std()),  4),
        "cv_r2_mean":   round(float(cv_r2.mean()),   4),
        "cv_r2_std":    round(float(cv_r2.std()),    4),
    }


# =============================================================================
# FIGURAS COMPARTIDAS
# =============================================================================

def plot_feature_importance(model, feature_cols, out_dir, filename, title):
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top = importances.nlargest(TOP_N_FEATURES).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("muted", len(top))
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel("Importancia (Mean Decrease Impurity)")
    ax.set_title(f"{title}\n(top {TOP_N_FEATURES} features)")
    plt.tight_layout()
    save(fig, out_dir, filename)


def plot_crossval(cv_acc, cv_f1m, cv_f1w, out_dir):
    folds = np.arange(1, CV_FOLDS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, scores, title, color in zip(
        axes,
        [cv_acc, cv_f1m, cv_f1w],
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

    plt.suptitle(f"Cross-validation ({CV_FOLDS} folds) — Clasificación {LABEL}",
                 y=1.02)
    plt.tight_layout()
    save(fig, out_dir, "fig_crossval_scores.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"Modelo Random Forest — {LABEL}"
    )
    parser.add_argument("--input",      required=True,
                        help="CSV limpio (aa_clean.csv)")
    parser.add_argument("--output_dir", default=f"results_{LABEL}/")
    args    = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  MODELO RANDOM FOREST — {LABEL}")
    print(f"{'='*55}")

    # Cargar y dividir
    (X, X_train, X_test,
     y_clf, y_clf_train, y_clf_test,
     y_reg, y_reg_train, y_reg_test,
     feature_cols) = load_and_split(Path(args.input))

    # Distribución de clases
    plot_class_distribution(y_clf, y_clf_train, y_clf_test, out_dir)

    # Clasificación
    clf         = train_classifier(X_train, y_clf_train)
    metrics_clf = evaluate_classifier(
        clf, X_train, X_test, y_clf_train, y_clf_test,
        X, y_clf, feature_cols, out_dir
    )

    # Regresión
    reg         = train_regressor(X_train, y_reg_train)
    metrics_reg = evaluate_regressor(
        reg, X_train, X_test, y_reg_train, y_reg_test,
        X, y_reg, feature_cols, out_dir
    )

    # Guardar modelos
    section("Guardando modelos")
    clf_path = out_dir / f"clf_{LABEL}.joblib"
    reg_path = out_dir / f"reg_{LABEL}.joblib"
    dump(clf, clf_path)
    dump(reg, reg_path)
    print(f"  → {clf_path.name}")
    print(f"  → {reg_path.name}")

    # Guardar métricas
    metrics = {
        "dataset":        LABEL,
        "n_train":        len(X_train),
        "n_test":         len(X_test),
        "n_features":     len(feature_cols),
        "test_size":      TEST_SIZE,
        "cv_folds":       CV_FOLDS,
        "rf_clf_params":  RF_CLF_PARAMS,
        "rf_reg_params":  RF_REG_PARAMS,
        "classification": metrics_clf,
        "regression":     metrics_reg,
    }
    metrics_path = out_dir / f"metrics_{LABEL}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → {metrics_path.name}")

    # Resumen final
    section("RESUMEN FINAL")
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

"""
learning_curves.py
------------------
Genera curvas de aprendizaje y validación para los modelos Random Forest
de clasificación (atomtype) y regresión (charge) para AA y UA.

Curvas generadas:
  1. Curva de aprendizaje — clasificación (accuracy vs tamaño de train)
  2. Curva de aprendizaje — regresión (R² vs tamaño de train)
  3. Curva de validación n_estimators — clasificación
  4. Curva de validación n_estimators — regresión
  5. Curva de validación max_depth — clasificación (bias/varianza)
  6. Curva de validación max_depth — regresión (bias/varianza)

Uso:
    python learning_curves.py --aa processed/aa_clean.csv
                               --ua processed/ua_clean.csv
                               --output_dir learning_curves/

Dependencias:
    pip install pandas numpy scikit-learn matplotlib seaborn
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (
    StratifiedKFold, learning_curve, validation_curve,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"]  = 11

# =============================================================================
# CONFIG
# =============================================================================

TARGET_CLF  = "atomtype"
TARGET_REG  = "charge"
RANDOM_SEED = 42
CV_FOLDS    = 5
MIN_SAMPLES = 2   # eliminar clases con menos instancias

RF_CLF_PARAMS = dict(
    n_estimators = 300,
    max_depth    = None,
    class_weight = "balanced",
    random_state = RANDOM_SEED,
    n_jobs       = -1,
)

RF_REG_PARAMS = dict(
    n_estimators = 300,
    max_depth    = None,
    random_state = RANDOM_SEED,
    n_jobs       = -1,
)

# Rango de n_estimators para curva de validación
N_ESTIMATORS_RANGE = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]

# Rango de max_depth para curva bias/varianza
MAX_DEPTH_RANGE = [1, 2, 3, 5, 7, 10, 15, 20, None]
MAX_DEPTH_LABELS = [str(d) if d is not None else "None"
                    for d in MAX_DEPTH_RANGE]


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


def load_data(csv_path: Path):
    """Carga dataset, elimina clases con 1 instancia y separa X, y_clf, y_reg."""
    df = pd.read_csv(csv_path)

    counts  = df[TARGET_CLF].value_counts()
    removed = sorted(counts[counts < MIN_SAMPLES].index.tolist())
    mask    = df[TARGET_CLF].isin(counts[counts >= MIN_SAMPLES].index)
    df      = df[mask].reset_index(drop=True)

    if removed:
        print(f"  Clases eliminadas (1 instancia): {removed}")

    feature_cols = [c for c in df.columns
                    if c not in [TARGET_CLF, TARGET_REG]]
    X     = df[feature_cols]
    y_clf = df[TARGET_CLF]
    y_reg = df[TARGET_REG]

    print(f"  {len(df)} átomos · {X.shape[1]} features · "
          f"{y_clf.nunique()} clases atomtype")
    return X, y_clf, y_reg


# =============================================================================
# 1 y 2 — CURVAS DE APRENDIZAJE
# =============================================================================

def plot_learning_curves(X, y_clf, y_reg, label, out_dir):
    """
    Curva de aprendizaje: rendimiento en train y validación en función
    del número de átomos usados para entrenar.

    Interpretación:
      - Si train y val convergen con pocos datos → el modelo generaliza bien.
      - Si val sigue subiendo al agregar datos → el modelo se beneficiaría
        de más moléculas en el dataset (alta varianza, underfitting).
      - Gap grande entre train y val → overfitting.
    """
    section(f"Curvas de aprendizaje — {label}")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                          random_state=RANDOM_SEED)

    # Tamaños de train a evaluar (de 10% a 100%)
    train_sizes = np.linspace(0.10, 1.0, 10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Clasificación ─────────────────────────────────────────────────────────
    print("  Calculando curva de aprendizaje — clasificación...")
    sizes_clf, train_scores_clf, val_scores_clf = learning_curve(
        RandomForestClassifier(**RF_CLF_PARAMS),
        X, y_clf,
        train_sizes = train_sizes,
        cv          = skf,
        scoring     = "accuracy",
        n_jobs      = -1,
    )

    train_mean_clf = train_scores_clf.mean(axis=1)
    train_std_clf  = train_scores_clf.std(axis=1)
    val_mean_clf   = val_scores_clf.mean(axis=1)
    val_std_clf    = val_scores_clf.std(axis=1)

    ax = axes[0]
    ax.plot(sizes_clf, train_mean_clf, "o-", color="#378ADD", label="Train")
    ax.fill_between(sizes_clf,
                    train_mean_clf - train_std_clf,
                    train_mean_clf + train_std_clf,
                    alpha=0.15, color="#378ADD")
    ax.plot(sizes_clf, val_mean_clf, "o-", color="#D85A30", label="Validación (CV)")
    ax.fill_between(sizes_clf,
                    val_mean_clf - val_std_clf,
                    val_mean_clf + val_std_clf,
                    alpha=0.15, color="#D85A30")
    ax.set_xlabel("Átomos en entrenamiento")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Curva de aprendizaje — Clasificación\n{label}")
    ax.set_ylim(0, 1.05)
    ax.legend()

    # ── Regresión ─────────────────────────────────────────────────────────────
    print("  Calculando curva de aprendizaje — regresión...")
    sizes_reg, train_scores_reg, val_scores_reg = learning_curve(
        RandomForestRegressor(**RF_REG_PARAMS),
        X, y_reg,
        train_sizes = train_sizes,
        cv          = CV_FOLDS,
        scoring     = "r2",
        n_jobs      = -1,
    )

    train_mean_reg = train_scores_reg.mean(axis=1)
    train_std_reg  = train_scores_reg.std(axis=1)
    val_mean_reg   = val_scores_reg.mean(axis=1)
    val_std_reg    = val_scores_reg.std(axis=1)

    ax = axes[1]
    ax.plot(sizes_reg, train_mean_reg, "o-", color="#378ADD", label="Train")
    ax.fill_between(sizes_reg,
                    train_mean_reg - train_std_reg,
                    train_mean_reg + train_std_reg,
                    alpha=0.15, color="#378ADD")
    ax.plot(sizes_reg, val_mean_reg, "o-", color="#D85A30", label="Validación (CV)")
    ax.fill_between(sizes_reg,
                    val_mean_reg - val_std_reg,
                    val_mean_reg + val_std_reg,
                    alpha=0.15, color="#D85A30")
    ax.set_xlabel("Átomos en entrenamiento")
    ax.set_ylabel("R²")
    ax.set_title(f"Curva de aprendizaje — Regresión\n{label}")
    ax.legend()

    plt.suptitle(f"Curvas de aprendizaje — {label}", y=1.02)
    plt.tight_layout()
    save(fig, out_dir, f"fig_learning_curves_{label}.png")


# =============================================================================
# 3 y 4 — CURVAS DE VALIDACIÓN: n_estimators
# =============================================================================

def plot_nestimators_curves(X, y_clf, y_reg, label, out_dir):
    """
    Curva de validación en función de n_estimators.

    Muestra a partir de cuántos árboles el modelo se estabiliza.
    Permite elegir el n_estimators óptimo sin desperdiciar cómputo.
    """
    section(f"Curva de validación n_estimators — {label}")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                          random_state=RANDOM_SEED)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Clasificación ─────────────────────────────────────────────────────────
    print("  Calculando curva n_estimators — clasificación...")
    clf_base = RandomForestClassifier(
        max_depth    = None,
        class_weight = "balanced",
        random_state = RANDOM_SEED,
        n_jobs       = -1,
    )
    train_scores, val_scores = validation_curve(
        clf_base, X, y_clf,
        param_name   = "n_estimators",
        param_range  = N_ESTIMATORS_RANGE,
        cv           = skf,
        scoring      = "accuracy",
        n_jobs       = -1,
    )

    ax = axes[0]
    ax.plot(N_ESTIMATORS_RANGE, train_scores.mean(axis=1),
            "o-", color="#378ADD", label="Train")
    ax.fill_between(N_ESTIMATORS_RANGE,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1),
                    alpha=0.15, color="#378ADD")
    ax.plot(N_ESTIMATORS_RANGE, val_scores.mean(axis=1),
            "o-", color="#D85A30", label="Validación (CV)")
    ax.fill_between(N_ESTIMATORS_RANGE,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1),
                    alpha=0.15, color="#D85A30")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Validación n_estimators — Clasificación\n{label}")
    ax.set_ylim(0, 1.05)
    ax.legend()

    # ── Regresión ─────────────────────────────────────────────────────────────
    print("  Calculando curva n_estimators — regresión...")
    reg_base = RandomForestRegressor(
        max_depth    = None,
        random_state = RANDOM_SEED,
        n_jobs       = -1,
    )
    train_scores_r, val_scores_r = validation_curve(
        reg_base, X, y_reg,
        param_name   = "n_estimators",
        param_range  = N_ESTIMATORS_RANGE,
        cv           = CV_FOLDS,
        scoring      = "r2",
        n_jobs       = -1,
    )

    ax = axes[1]
    ax.plot(N_ESTIMATORS_RANGE, train_scores_r.mean(axis=1),
            "o-", color="#378ADD", label="Train")
    ax.fill_between(N_ESTIMATORS_RANGE,
                    train_scores_r.mean(axis=1) - train_scores_r.std(axis=1),
                    train_scores_r.mean(axis=1) + train_scores_r.std(axis=1),
                    alpha=0.15, color="#378ADD")
    ax.plot(N_ESTIMATORS_RANGE, val_scores_r.mean(axis=1),
            "o-", color="#D85A30", label="Validación (CV)")
    ax.fill_between(N_ESTIMATORS_RANGE,
                    val_scores_r.mean(axis=1) - val_scores_r.std(axis=1),
                    val_scores_r.mean(axis=1) + val_scores_r.std(axis=1),
                    alpha=0.15, color="#D85A30")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("R²")
    ax.set_title(f"Validación n_estimators — Regresión\n{label}")
    ax.legend()

    plt.suptitle(f"Curva de validación n_estimators — {label}", y=1.02)
    plt.tight_layout()
    save(fig, out_dir, f"fig_nestimators_{label}.png")


# =============================================================================
# 5 y 6 — CURVAS DE VALIDACIÓN: max_depth (bias/varianza)
# =============================================================================

def plot_depth_curves(X, y_clf, y_reg, label, out_dir):
    """
    Curva de validación en función de max_depth.

    Muestra el tradeoff bias/varianza:
      - max_depth pequeño → underfitting (alto bias, bajo train score)
      - max_depth grande → overfitting (gap grande entre train y val)
      - max_depth óptimo → donde val score es máximo

    Para Random Forest con max_depth=None los árboles crecen completos.
    Esta curva justifica esa elección o sugiere una profundidad óptima.
    """
    section(f"Curva de validación max_depth (bias/varianza) — {label}")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                          random_state=RANDOM_SEED)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Clasificación ─────────────────────────────────────────────────────────
    print("  Calculando curva max_depth — clasificación...")
    clf_base = RandomForestClassifier(
        n_estimators = 300,
        class_weight = "balanced",
        random_state = RANDOM_SEED,
        n_jobs       = -1,
    )
    train_scores, val_scores = validation_curve(
        clf_base, X, y_clf,
        param_name   = "max_depth",
        param_range  = MAX_DEPTH_RANGE,
        cv           = skf,
        scoring      = "accuracy",
        n_jobs       = -1,
    )

    x_ticks = range(len(MAX_DEPTH_LABELS))
    ax = axes[0]
    ax.plot(x_ticks, train_scores.mean(axis=1),
            "o-", color="#378ADD", label="Train")
    ax.fill_between(x_ticks,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1),
                    alpha=0.15, color="#378ADD")
    ax.plot(x_ticks, val_scores.mean(axis=1),
            "o-", color="#D85A30", label="Validación (CV)")
    ax.fill_between(x_ticks,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1),
                    alpha=0.15, color="#D85A30")
    ax.set_xticks(list(x_ticks))
    ax.set_xticklabels(MAX_DEPTH_LABELS)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Bias/Varianza max_depth — Clasificación\n{label}")
    ax.set_ylim(0, 1.05)
    ax.legend()

    # ── Regresión ─────────────────────────────────────────────────────────────
    print("  Calculando curva max_depth — regresión...")
    reg_base = RandomForestRegressor(
        n_estimators = 300,
        random_state = RANDOM_SEED,
        n_jobs       = -1,
    )
    train_scores_r, val_scores_r = validation_curve(
        reg_base, X, y_reg,
        param_name   = "max_depth",
        param_range  = MAX_DEPTH_RANGE,
        cv           = CV_FOLDS,
        scoring      = "r2",
        n_jobs       = -1,
    )

    ax = axes[1]
    ax.plot(x_ticks, train_scores_r.mean(axis=1),
            "o-", color="#378ADD", label="Train")
    ax.fill_between(x_ticks,
                    train_scores_r.mean(axis=1) - train_scores_r.std(axis=1),
                    train_scores_r.mean(axis=1) + train_scores_r.std(axis=1),
                    alpha=0.15, color="#378ADD")
    ax.plot(x_ticks, val_scores_r.mean(axis=1),
            "o-", color="#D85A30", label="Validación (CV)")
    ax.fill_between(x_ticks,
                    val_scores_r.mean(axis=1) - val_scores_r.std(axis=1),
                    val_scores_r.mean(axis=1) + val_scores_r.std(axis=1),
                    alpha=0.15, color="#D85A30")
    ax.set_xticks(list(x_ticks))
    ax.set_xticklabels(MAX_DEPTH_LABELS)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("R²")
    ax.set_title(f"Bias/Varianza max_depth — Regresión\n{label}")
    ax.legend()

    plt.suptitle(f"Curva de validación max_depth — {label}", y=1.02)
    plt.tight_layout()
    save(fig, out_dir, f"fig_depth_bias_variance_{label}.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Curvas de aprendizaje y validación — AA y UA"
    )
    parser.add_argument("--aa",         required=True, help="aa_clean.csv")
    parser.add_argument("--ua",         required=True, help="ua_clean.csv")
    parser.add_argument("--output_dir", default="learning_curves/")
    args    = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path, label in [(args.aa, "AA"), (args.ua, "UA")]:
        print(f"\n{'='*55}")
        print(f"  DATASET: {label}")
        print(f"{'='*55}")

        X, y_clf, y_reg = load_data(Path(csv_path))

        plot_learning_curves(X, y_clf, y_reg, label, out_dir)
        plot_nestimators_curves(X, y_clf, y_reg, label, out_dir)
        plot_depth_curves(X, y_clf, y_reg, label, out_dir)

    print(f"\n✔ Figuras guardadas en: {out_dir}")
    print("  Archivos generados:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  → {f.name}")


if __name__ == "__main__":
    main()
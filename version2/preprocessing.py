"""
preprocessing.py
----------------
Preprocesamiento de los datasets All-Atom (AA) y United-Atom (UA)
para entrenamiento de modelos Random Forest.

Transformaciones aplicadas:
  1. Eliminación de átomos erróneos
  2. Eliminación de features redundantes
  3. Encoding de variables booleanas
  4. Encoding one-hot de variables categóricas simples
  5. Feature hashing de variables categóricas de alta cardinalidad
  6. Separación de columnas de identificación

Nota sobre normalización:
  Random Forest es invariante a la escala de las features — no requiere
  StandardScaler ni MinMaxScaler. Las features numéricas se mantienen
  en su escala original.

Nota sobre clases raras:
  Se mantienen todas las clases, incluyendo las de baja frecuencia.
  El desbalance se maneja durante el entrenamiento con class_weight='balanced'.

Uso:
    python preprocessing.py --aa dataset_AA.csv --ua dataset_UA.csv --output_dir processed/

Salida:
    processed/aa_clean.csv
    processed/ua_clean.csv
    processed/aa_ids.csv
    processed/ua_ids.csv
    processed/preprocessing_artifacts.json
    processed/preprocessing_report.txt

Dependencias:
    pip install pandas numpy scikit-learn
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher


# =============================================================================
# CONFIG
# =============================================================================

# Columnas que identifican al átomo pero no son features del modelo
ID_COLS = ["molecule", "atom_id", "atom_name", "bonded_to_element"]

# Targets — no se transforman, se guardan tal cual
TARGET_COLS = ["atomtype", "charge"]

# Features redundantes detectadas en el EDA:
#   degree_of_unsat ↔ bonds_aromatic:  r = 1.000
#   bonds_single    ↔ coordination:    r = 0.979
#   n_dihedrals     ↔ coordination:    r = 0.854
COLS_TO_DROP = ["degree_of_unsat", "n_dihedrals", "bonds_single","tripos_type"]

# Variables booleanas → 0/1
BOOL_COLS = ["is_planar", "is_in_ring", "is_aromatic"]

# Variables categóricas simples → one-hot
CAT_COLS = ["element"]

# Variables categóricas de alta cardinalidad → feature hashing
HASH_COLS = ["neighbor_config", "neighbor2_config"]
HASH_N_FEATURES = 32

# Átomo erróneo identificado en el EDA (UA/2NP: element=O con atomtype=HC)
ERRONEOUS_ATOMS = [
    {"dataset": "UA", "molecule": "2NP", "element": "O", "atomtype": "HC"},
]


# =============================================================================
# HELPERS
# =============================================================================

def log(msg, report_lines):
    print(msg)
    report_lines.append(msg)


def section(title, report_lines):
    log(f"\n{'='*55}\n  {title}\n{'='*55}", report_lines)


# =============================================================================
# PASOS DEL PIPELINE
# =============================================================================

def remove_erroneous_atoms(df, dataset_label, report_lines):
    """
    Elimina átomos con asignación incorrecta de atomtype detectados en el EDA.

    El átomo erróneo en UA (2NP, element=O, atomtype=HC) es producto de un
    error de match entre el mol2 y el itp de 2NP, cuyo mol2 tiene atom_names
    genéricos que no permiten un match unívoco por nombre. Al hacer el match
    por índice posicional, este átomo recibió un atomtype incorrecto.
    """
    for err in ERRONEOUS_ATOMS:
        if err["dataset"] != dataset_label:
            continue
        mask = (
            (df["molecule"] == err["molecule"]) &
            (df["element"]  == err["element"])  &
            (df["atomtype"] == err["atomtype"])
        )
        n = mask.sum()
        df = df[~mask].copy()
        log(f"  Eliminado ({err['molecule']}, {err['element']}, "
            f"{err['atomtype']}): {n} átomo(s)", report_lines)
    return df


def drop_redundant_features(df, report_lines):
    """
    Elimina features redundantes detectadas en el análisis de correlación.

    Se conserva en cada par la feature más interpretable:
      bonds_aromatic  sobre degree_of_unsat  (r = 1.000)
      coordination    sobre bonds_single     (r = 0.979)
      coordination    sobre n_dihedrals      (r = 0.854)
    """
    cols = [c for c in COLS_TO_DROP if c in df.columns]
    df   = df.drop(columns=cols)
    log(f"  Eliminadas: {cols}", report_lines)
    return df


def encode_booleans(df, report_lines):
    """Convierte True/False a 1/0 sin pérdida de información."""
    cols = [c for c in BOOL_COLS if c in df.columns]
    df[cols] = df[cols].astype(int)
    log(f"  Booleanas → 0/1: {cols}", report_lines)
    return df


def encode_categorical(df, report_lines):
    """
    One-hot encoding para element y tripos_type.

    Preferible a label encoding para Random Forest porque no introduce
    un orden artificial entre categorías sin sentido químico.
    Devuelve (df_encoded, ohe_categories) donde ohe_categories guarda
    qué categorías existían al momento del encoding — necesario para
    aplicar el mismo encoding a nuevas moléculas.
    """
    cols = [c for c in CAT_COLS if c in df.columns]
    ohe_categories = {col: sorted(df[col].dropna().unique().tolist())
                      for col in cols}
    df = pd.get_dummies(df, columns=cols, prefix=cols, dtype=int)
    new_cols = [c for c in df.columns
                if any(c.startswith(f"{cat}_") for cat in cols)]
    log(f"  One-hot: {cols} → {len(new_cols)} columnas", report_lines)
    return df, ohe_categories


def encode_high_cardinality(df, report_lines):
    """
    Feature hashing para neighbor_config y neighbor2_config.

    Con 32-53 valores únicos, one-hot generaría demasiadas columnas y
    no generalizaría a configuraciones nuevas. El hashing proyecta cada
    string a un vector de dimensión fija (32 componentes) usando una
    función hash, permitiendo generalizar a moléculas no vistas.
    """
    for col in HASH_COLS:
        if col not in df.columns:
            continue
        hasher    = FeatureHasher(n_features=HASH_N_FEATURES, input_type="string")
        hashed    = hasher.transform(
            df[col].fillna("").apply(lambda x: [x])
        ).toarray()
        col_names = [f"{col}_h{i}" for i in range(HASH_N_FEATURES)]
        hashed_df = pd.DataFrame(hashed, columns=col_names,
                                 index=df.index, dtype=int)
        df = pd.concat([df.drop(columns=[col]), hashed_df], axis=1)
        log(f"  Hashing '{col}' → {HASH_N_FEATURES} componentes", report_lines)
    return df


def separate_id_columns(df, report_lines):
    """
    Separa las columnas de identificación del input del modelo.

    molecule, atom_id, atom_name y bonded_to_element identifican al átomo
    pero no son features predictivas. Se guardan separadas para poder
    rastrear predicciones por átomo durante la evaluación.
    """
    id_cols = [c for c in ID_COLS if c in df.columns]
    df_ids  = df[id_cols].copy()
    df      = df.drop(columns=id_cols)
    log(f"  ID cols separadas: {id_cols}", report_lines)
    return df, df_ids


# =============================================================================
# ARTIFACTS
# =============================================================================

def build_artifacts(label, df_original, df_clean, ohe_categories):
    """
    Construye el diccionario de artifacts para este dataset.

    Contiene toda la información necesaria para:
      - Reproducir exactamente el mismo preprocesamiento en nuevas moléculas.
      - Saber qué categorías existían en el one-hot (para revertir o extender).
      - Rastrear qué se eliminó y por qué.
    """
    return {
        "dataset":          label,
        "original_shape":   list(df_original.shape),
        "clean_shape":      list(df_clean.shape),
        "n_atoms":          len(df_clean),
        "n_atomtypes":      int(df_clean["atomtype"].nunique()),
        "atomtypes":        sorted(df_clean["atomtype"].unique().tolist()),
        "charge_range":     [float(df_clean["charge"].min()),
                             float(df_clean["charge"].max())],
        "dropped_features": COLS_TO_DROP,
        "bool_cols":        BOOL_COLS,
        "id_cols":          ID_COLS,
        "hash_cols":        {col: HASH_N_FEATURES for col in HASH_COLS},
        "ohe_categories":   ohe_categories,
        "erroneous_atoms_removed": [
            e for e in ERRONEOUS_ATOMS if e["dataset"] == label
        ],
    }


# =============================================================================
# PIPELINE COMPLETO
# =============================================================================

def preprocess(df, dataset_label, report_lines):
    """Aplica el pipeline completo. Devuelve (df_clean, df_ids, artifacts)."""

    df_original = df.copy()
    section(f"PREPROCESAMIENTO — {dataset_label}", report_lines)
    log(f"  Shape inicial: {df.shape}", report_lines)

    log("\n  [1] Eliminando átomos erróneos...", report_lines)
    df = remove_erroneous_atoms(df, dataset_label, report_lines)

    log("\n  [2] Eliminando features redundantes...", report_lines)
    df = drop_redundant_features(df, report_lines)

    log("\n  [3] Encoding de booleanas...", report_lines)
    df = encode_booleans(df, report_lines)

    log("\n  [4] One-hot encoding...", report_lines)
    df, ohe_categories = encode_categorical(df, report_lines)

    log("\n  [5] Feature hashing...", report_lines)
    df = encode_high_cardinality(df, report_lines)

    log("\n  [6] Separando columnas de identificación...", report_lines)
    df, df_ids = separate_id_columns(df, report_lines)

    log(f"\n  Shape final: {df.shape}", report_lines)
    log(f"  Features: {df.shape[1] - len(TARGET_COLS)}", report_lines)

    nulls = df.isnull().sum().sum()
    log(f"  Valores nulos: {nulls} {'✓' if nulls == 0 else '⚠'}", report_lines)
    log(f"  Clases atomtype: {df['atomtype'].nunique()}", report_lines)
    log(f"  Rango charge: [{df['charge'].min():.3f}, {df['charge'].max():.3f}]",
        report_lines)

    artifacts = build_artifacts(dataset_label, df_original, df, ohe_categories)
    return df, df_ids, artifacts


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocesamiento AA y UA para Random Forest"
    )
    parser.add_argument("--aa",         required=True, help="CSV All-Atom")
    parser.add_argument("--ua",         required=True, help="CSV United-Atom")
    parser.add_argument("--output_dir", default="processed/")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_lines = ["REPORTE DE PREPROCESAMIENTO", "=" * 55]

    df_AA = pd.read_csv(args.aa)
    df_UA = pd.read_csv(args.ua)
    print(f"AA cargado: {df_AA.shape}")
    print(f"UA cargado: {df_UA.shape}")

    df_AA_clean, df_AA_ids, artifacts_AA = preprocess(df_AA, "AA", report_lines)
    df_UA_clean, df_UA_ids, artifacts_UA = preprocess(df_UA, "UA", report_lines)

    # Datasets limpios e IDs
    df_AA_clean.to_csv(out_dir / "aa_clean.csv", index=False)
    df_UA_clean.to_csv(out_dir / "ua_clean.csv", index=False)
    df_AA_ids.to_csv(out_dir   / "aa_ids.csv",   index=False)
    df_UA_ids.to_csv(out_dir   / "ua_ids.csv",   index=False)

    # Un único archivo de artifacts con AA y UA juntos
    with open(out_dir / "preprocessing_artifacts.json", "w") as f:
        json.dump({"AA": artifacts_AA, "UA": artifacts_UA}, f, indent=2)

    # Reporte
    with open(out_dir / "preprocessing_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n✔ Archivos guardados en: {out_dir}")
    print(f"  aa_clean.csv                 → {df_AA_clean.shape}")
    print(f"  ua_clean.csv                 → {df_UA_clean.shape}")
    print(f"  aa_ids.csv / ua_ids.csv")
    print(f"  preprocessing_artifacts.json → mapa completo AA + UA")
    print(f"  preprocessing_report.txt")


if __name__ == "__main__":
    main()

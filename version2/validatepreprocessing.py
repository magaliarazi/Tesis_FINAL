"""
validate_preprocessing.py
--------------------------
Valida que los datasets preprocesados AA y UA estén correctos y sean
compatibles entre sí para el entrenamiento de modelos Random Forest.

Checks realizados:
  1. Archivos existen y se pueden leer
  2. Sin valores nulos
  3. Targets presentes (atomtype, charge)
  4. Sin átomos erróneos conocidos
  5. Columnas idénticas en AA y UA
  6. Shapes razonables
  7. Artifacts JSON consistente con los CSVs

Uso:
    python validatepreprocessing.py --processed_dir processed/

Dependencias:
    pip install pandas numpy
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np


# =============================================================================
# HELPERS
# =============================================================================

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

def check(condition, msg_ok, msg_fail, warnings):
    if condition:
        print(f"{PASS} {msg_ok}")
        return True
    else:
        print(f"{FAIL} {msg_fail}")
        warnings.append(msg_fail)
        return False


def section(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")


# =============================================================================
# CHECKS INDIVIDUALES
# =============================================================================

def check_files_exist(processed_dir: Path, warnings: list) -> bool:
    section("1. Archivos esperados")
    required = [
        "aa_clean.csv", "ua_clean.csv",
        "aa_ids.csv",   "ua_ids.csv",
        "preprocessing_artifacts.json",
        "preprocessing_report.txt",
    ]
    all_ok = True
    for f in required:
        exists = (processed_dir / f).exists()
        check(exists, f"{f} existe", f"{f} NO encontrado", warnings)
        if not exists:
            all_ok = False
    return all_ok


def check_no_nulls(df: pd.DataFrame, label: str, warnings: list):
    section(f"2. Valores nulos — {label}")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print(f"{PASS} Sin valores nulos")
    else:
        for col, n in nulls.items():
            msg = f"{col}: {n} nulos"
            print(f"{FAIL} {msg}")
            warnings.append(f"{label} — {msg}")


def check_targets(df: pd.DataFrame, label: str, warnings: list):
    section(f"3. Targets — {label}")
    for target in ["atomtype", "charge"]:
        check(target in df.columns,
              f"'{target}' presente",
              f"'{target}' NO encontrado en {label}",
              warnings)

    if "atomtype" in df.columns:
        print(f"  Clases de atomtype: {df['atomtype'].nunique()}")
        print(f"  {df['atomtype'].value_counts().to_dict()}")

    if "charge" in df.columns:
        print(f"  Rango charge: [{df['charge'].min():.3f}, {df['charge'].max():.3f}]")
        print(f"  Media: {df['charge'].mean():.4f} | Std: {df['charge'].std():.4f}")


def check_no_erroneous_atoms(df: pd.DataFrame, label: str, warnings: list):
    section(f"4. Átomos erróneos conocidos — {label}")
    # H con element=O
    bad_hc = df[(df.get("element", pd.Series()) == "O") &
                (df.get("atomtype", pd.Series()) == "HC")]
    check(len(bad_hc) == 0,
          "Sin átomos O con atomtype HC",
          f"{len(bad_hc)} átomo(s) O con atomtype HC en {label}",
          warnings)


def check_column_alignment(df_AA: pd.DataFrame, df_UA: pd.DataFrame,
                            warnings: list):
    section("5. Alineación de columnas AA vs UA")

    cols_AA = set(df_AA.columns)
    cols_UA = set(df_UA.columns)

    only_AA = cols_AA - cols_UA
    only_UA = cols_UA - cols_AA

    check(len(only_AA) == 0,
          "Sin columnas exclusivas de AA",
          f"Columnas solo en AA ({len(only_AA)}): {sorted(only_AA)}",
          warnings)

    check(len(only_UA) == 0,
          "Sin columnas exclusivas de UA",
          f"Columnas solo en UA ({len(only_UA)}): {sorted(only_UA)}",
          warnings)

    if only_AA or only_UA:
        print(f"\n  {WARN} Las columnas difieren entre AA y UA.")
        print("  Esto puede causar problemas al aplicar el modelo a nuevas moléculas.")
        print("  Solución: usar pd.get_dummies con un índice de categorías fijo")
        print("  (guardado en preprocessing_artifacts.json → ohe_categories).")
    else:
        check(list(df_AA.columns) == list(df_UA.columns),
              "Columnas en el mismo orden",
              "Columnas en distinto orden — reindexar antes de entrenar",
              warnings)


def check_shapes(df_AA: pd.DataFrame, df_UA: pd.DataFrame,
                 df_AA_ids: pd.DataFrame, df_UA_ids: pd.DataFrame,
                 warnings: list):
    section("6. Shapes")

    print(f"  AA clean:  {df_AA.shape}")
    print(f"  UA clean:  {df_UA.shape}")
    print(f"  AA ids:    {df_AA_ids.shape}")
    print(f"  UA ids:    {df_UA_ids.shape}")

    check(df_AA.shape[1] == df_UA.shape[1],
          "AA y UA tienen el mismo número de columnas",
          f"Distinto número de columnas: AA={df_AA.shape[1]}, UA={df_UA.shape[1]}",
          warnings)

    check(len(df_AA) == len(df_AA_ids),
          "AA clean y AA ids tienen el mismo número de filas",
          f"Filas no coinciden: clean={len(df_AA)}, ids={len(df_AA_ids)}",
          warnings)

    check(len(df_UA) == len(df_UA_ids),
          "UA clean y UA ids tienen el mismo número de filas",
          f"Filas no coinciden: clean={len(df_UA)}, ids={len(df_UA_ids)}",
          warnings)

    # UA debe tener menos átomos que AA (sin H apolares)
    check(len(df_UA) < len(df_AA),
          f"UA ({len(df_UA)}) tiene menos átomos que AA ({len(df_AA)}) ✓",
          f"UA ({len(df_UA)}) debería tener menos átomos que AA ({len(df_AA)})",
          warnings)


def check_artifacts(processed_dir: Path,
                    df_AA: pd.DataFrame, df_UA: pd.DataFrame,
                    warnings: list):
    section("7. Consistencia con artifacts.json")

    path = processed_dir / "preprocessing_artifacts.json"
    if not path.exists():
        print(f"{FAIL} preprocessing_artifacts.json no encontrado")
        warnings.append("preprocessing_artifacts.json no encontrado")
        return

    with open(path) as f:
        artifacts = json.load(f)

    for label, df in [("AA", df_AA), ("UA", df_UA)]:
        art = artifacts.get(label, {})
        if not art:
            print(f"{FAIL} Sin artifacts para {label}")
            warnings.append(f"Sin artifacts para {label}")
            continue

        # Shape final
        expected_shape = art.get("clean_shape", [])
        check(list(df.shape) == expected_shape,
              f"{label}: shape coincide con artifacts {expected_shape}",
              f"{label}: shape {list(df.shape)} ≠ artifacts {expected_shape}",
              warnings)

        # Atomtypes
        expected_types = set(art.get("atomtypes", []))
        actual_types   = set(df["atomtype"].unique()) if "atomtype" in df.columns else set()
        check(expected_types == actual_types,
              f"{label}: atomtypes coinciden con artifacts",
              f"{label}: atomtypes no coinciden — "
              f"extra: {actual_types - expected_types}, "
              f"faltantes: {expected_types - actual_types}",
              warnings)

        # ohe_categories
        ohe = art.get("ohe_categories", {})
        for col, cats in ohe.items():
            expected_cols = {f"{col}_{cat}" for cat in cats}
            actual_cols   = {c for c in df.columns if c.startswith(f"{col}_")}
            check(expected_cols == actual_cols,
                  f"{label}: columnas one-hot de '{col}' coinciden",
                  f"{label}: columnas one-hot de '{col}' no coinciden — "
                  f"extra: {actual_cols - expected_cols}, "
                  f"faltantes: {expected_cols - actual_cols}",
                  warnings)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Valida los datasets preprocesados AA y UA"
    )
    parser.add_argument("--processed_dir", default="processed/",
                        help="Carpeta con los archivos preprocesados")
    args   = parser.parse_args()
    out    = Path(args.processed_dir)

    print("\n" + "="*55)
    print("  VALIDACIÓN DEL PREPROCESAMIENTO")
    print("="*55)
    print(f"  Directorio: {out.resolve()}")

    warnings = []

    # Check 1 — archivos existen
    if not check_files_exist(out, warnings):
        print(f"\n{FAIL} Faltan archivos críticos. Corré preprocessing.py primero.")
        return

    # Cargar
    df_AA     = pd.read_csv(out / "aa_clean.csv")
    df_UA     = pd.read_csv(out / "ua_clean.csv")
    df_AA_ids = pd.read_csv(out / "aa_ids.csv")
    df_UA_ids = pd.read_csv(out / "ua_ids.csv")

    # Checks
    check_no_nulls(df_AA, "AA", warnings)
    check_no_nulls(df_UA, "UA", warnings)
    check_targets(df_AA, "AA", warnings)
    check_targets(df_UA, "UA", warnings)
    check_no_erroneous_atoms(df_AA, "AA", warnings)
    check_no_erroneous_atoms(df_UA, "UA", warnings)
    check_column_alignment(df_AA, df_UA, warnings)
    check_shapes(df_AA, df_UA, df_AA_ids, df_UA_ids, warnings)
    check_artifacts(out, df_AA, df_UA, warnings)

    # Resumen final
    section("RESUMEN FINAL")
    if not warnings:
        print(f"{PASS} Todos los checks pasaron. Los datasets están listos para entrenar.")
    else:
        print(f"{FAIL} Se encontraron {len(warnings)} problema(s):\n")
        for i, w in enumerate(warnings, 1):
            print(f"  {i}. {w}")
        print("\n  Revisá los errores antes de continuar con el modelo.")


if __name__ == "__main__":
    main()
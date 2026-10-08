"""
split_AA_UA_v2.py
-----------------
Divide el dataset en dos tablas: All-Atom (AA) y United-Atom (UA).

Reglas (sin hardcodear atomtypes):

  TABLA AA:
    - Todo átomo que NO es C ni H  →  va
    - C con H_count == 0 (C desnudo)  →  va
    - C con H_count > 0  →  va (en AA los H son explícitos)
    - H unido a C (bonded_to_element == "C")  →  va
    - H polar (bonded_to_element en {N, O})  →  va

  TABLA UA:
    - Todo átomo que NO es C ni H  →  va
    - C con H_count == 0 (C desnudo)  →  va
    - C con H_count > 0 (united-atom, absorbe sus H)  →  va
    - H polar (bonded_to_element en {N, O})  →  va
    - H unido a C  →  NO va (absorbido en el C)

Requiere columna 'bonded_to_element' generada por extract_dataset_v2.py.

Uso:
    python split_v2.py --input dataset_remapped.csv --output_AA dataset_AA.csv --output_UA dataset_UA.csv
"""

import argparse
import pandas as pd

POLAR_BOND_ELEMENTS = {"N", "O"}


def is_AA(row):
    elem = row["element"]
    bonded = row["bonded_to_element"]

    # todo lo que no es C ni H va a AA
    if elem not in ("C", "H"):
        return True

    # C → siempre va a AA (desnudo o con H explícitos)
    if elem == "C":
        return True

    # H unido a C → va a AA
    if elem == "H" and bonded == "C":
        return True

    # H polar (unido a N u O) → va a AA
    if elem == "H" and bonded in POLAR_BOND_ELEMENTS:
        return True

    return False


def is_UA(row):
    elem = row["element"]
    bonded = row["bonded_to_element"]

    # todo lo que no es C ni H va a UA
    if elem not in ("C", "H"):
        return True

    # C → siempre va a UA (desnudo o united-atom)
    if elem == "C":
        return True

    # H polar (unido a N u O) → va a UA
    if elem == "H" and bonded in POLAR_BOND_ELEMENTS:
        return True

    # H unido a C → NO va a UA
    return False


def split(input_path, output_AA, output_UA):
    df = pd.read_csv(input_path)
    print(f"\n📥 Dataset original: {df.shape}")

    if "bonded_to_element" not in df.columns:
        raise ValueError(
            "Falta la columna 'bonded_to_element'. "
            "Regenerá el dataset con extract_dataset_v2.py"
        )

    df_AA = df[df.apply(is_AA, axis=1)].copy()
    df_UA = df[df.apply(is_UA, axis=1)].copy()

    df_AA.to_csv(output_AA, index=False)
    df_UA.to_csv(output_UA, index=False)

    # ── resumen ───────────────────────────────────────────────────────────────
    print(f"\n📊 AA: {df_AA.shape[0]} átomos")
    print(f"   elementos: {df_AA['element'].value_counts().to_dict()}")

    print(f"\n📊 UA: {df_UA.shape[0]} átomos")
    print(f"   elementos: {df_UA['element'].value_counts().to_dict()}")

    hc_in_ua = ((df_UA["element"] == "H") & (df_UA["bonded_to_element"] == "C")).sum()
    print(f"\n✅ H unidos a C en UA: {hc_in_ua} (debe ser 0)")

    print(f"\n💾 Guardados:\n  {output_AA}\n  {output_UA}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     "-i",  required=True)
    parser.add_argument("--output_AA", required=True)
    parser.add_argument("--output_UA", required=True)
    args = parser.parse_args()

    split(args.input, args.output_AA, args.output_UA)
"""
remap_atomtypes.py
------------------
Aplica el remapeo de atomtypes sobre el dataset original antes del split
AA/UA, consolidando clases raras o equivalentes según el criterio de la
tutora.

Remapeos aplicados:
  NOpt  → N      (NOpt es equivalente a N)
  NPri  → N      (NPri es equivalente a N)
  CLAro → CL     (CLAro es una creación del ATB, equivale a CL)
  SDmso → S      (SDmso tiene muy pocas instancias, se generaliza a S)
  OM    → OM     (se mantiene, 4 instancias, químicamente distinto de OE)

Justificación química:
  - NOpt y NPri son tipos de nitrógeno cuya distinción es una convención
    del ATB que no refleja diferencias en el campo de fuerzas original.
  - CLAro es una creación del ATB para cloro aromático; en el campo de
    fuerzas original ambos cloros se tratan como CL.
  - SDmso tiene solo 1 instancia (azufre en dimetilsulfóxido), insuficiente
    para entrenar. Se generaliza a S como tipo atómico de azufre genérico.
  - OM (oxígeno del grupo éster) se mantiene separado de OE (oxígeno éter)
    por ser químicamente distinto y tener 4 instancias — manejable con
    class_weight='balanced'.

Uso:
    python remap_atomtypes.py --input dataset.csv --output dataset_remapped.csv

Dependencias:
    pip install pandas
"""

import argparse
from pathlib import Path

import pandas as pd


# =============================================================================
# REMAPEO
# =============================================================================

ATOMTYPE_REMAP = {
    "NOpt":  "N",
    "NPri":  "N",
    "CLAro": "CL",
    "SDmso": "S",
    "CAro":  "C",
}

# OM se mantiene — 4 instancias, químicamente distinto de OE
# OE se mantiene
# F  se mantiene — aunque pocas instancias, relevante en diseño de fármacos


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Remapeo de atomtypes raros o equivalentes"
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="dataset.csv generado por extract_dataset_v4.py")
    parser.add_argument("--output", "-o", default="dataset_remapped.csv",
                        help="CSV de salida con atomtypes remapeados")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"\n📥 Dataset original: {df.shape}")
    print(f"   Clases atomtype: {df['atomtype'].nunique()}")
    print(f"\n   Distribución original:")
    print(df["atomtype"].value_counts().to_string())

    # Aplicar remapeo
    df["atomtype"] = df["atomtype"].replace(ATOMTYPE_REMAP)

    print(f"\n✔ Remapeo aplicado:")
    for original, nuevo in ATOMTYPE_REMAP.items():
        print(f"   {original} → {nuevo}")

    print(f"\n   Distribución después del remapeo:")
    print(df["atomtype"].value_counts().to_string())
    print(f"\n   Clases atomtype después: {df['atomtype'].nunique()}")

    # Guardar
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Guardado en: {output_path}")
    print(f"   {len(df)} filas · {df.shape[1]} columnas")


if __name__ == "__main__":
    main()
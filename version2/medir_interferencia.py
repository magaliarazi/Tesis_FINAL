"""
medir_tiempo_inferencia.py
---------------------------
Mide tiempo de inferencia real (wall-clock) de los modelos RF y TabTransformer
sobre moléculas de distinto tamaño (N átomos), usando el pipeline real:

    processed/aa_clean.csv + processed/aa_ids.csv   (o ua_*)
    results_AA/clf_AA.joblib, results_AA/reg_AA.joblib
    tabtransformer_best.pt (checkpoint guardado por tabtransformer_atomtype_charge.py)

Genera un CSV (tiempos_inferencia.csv) con columnas:
    modelo, molecula, n_atomos, tiempo_medio_s, tiempo_std_s

Uso:
    python medir_tiempo_inferencia.py \
        --clean_csv processed/aa_clean.csv \
        --ids_csv   processed/aa_ids.csv \
        --clf_path  results_AA/clf_AA.joblib \
        --reg_path  results_AA/reg_AA.joblib \
        --tabt_ckpt tabtransformer_best.pt \
        --n_repeticiones 20
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

# Debe estar disponible en el mismo entorno / carpeta que
# tabtransformer_atomtype_charge.py, para reconstruir la arquitectura
from TabTransformer import (
    ALL_FEATURE_COLS, TabTransformerModel,
)

TARGET_CLF = "atomtype"
TARGET_REG = "charge"


# =============================================================================
# CARGA DE DATOS + SELECCIÓN DE MOLÉCULAS DE DISTINTO TAMAÑO
# =============================================================================

def cargar_moleculas(clean_csv: Path, ids_csv: Path, n_moleculas: int = 5):
    """
    Reconstruye por molécula el conjunto de features (mismo formato que
    entra al predict() del RF), agrupando aa_clean.csv por molécula usando
    aa_ids.csv (mismo orden de filas, según separate_id_columns en
    preprocessing.py).
    """
    df_clean = pd.read_csv(clean_csv)
    df_ids   = pd.read_csv(ids_csv)

    assert len(df_clean) == len(df_ids), \
        "clean_csv e ids_csv deben tener el mismo número de filas"

    df_full = pd.concat([df_ids.reset_index(drop=True),
                          df_clean.reset_index(drop=True)], axis=1)

    feature_cols = [c for c in df_clean.columns
                    if c not in (TARGET_CLF, TARGET_REG)]

    # Tamaño (n_atomos) por molécula
    tam_por_molecula = df_full.groupby("molecule").size().sort_values()

    # Elegir n_moleculas repartidas a lo largo del rango de tamaños
    idx = np.linspace(0, len(tam_por_molecula) - 1, n_moleculas).astype(int)
    seleccionadas = tam_por_molecula.iloc[idx]

    moleculas = {}
    for nombre, n_atomos in seleccionadas.items():
        sub = df_full[df_full["molecule"] == nombre]
        X = sub[feature_cols].reset_index(drop=True)
        moleculas[nombre] = (int(n_atomos), X)

    return moleculas, feature_cols


# =============================================================================
# MEDICIÓN — RANDOM FOREST
# =============================================================================

def medir_rf(modelo, X: pd.DataFrame, n_repeticiones: int):
    _ = modelo.predict(X)  # warm-up, no se cuenta

    tiempos = []
    for _ in range(n_repeticiones):
        t0 = time.perf_counter()
        _ = modelo.predict(X)
        t1 = time.perf_counter()
        tiempos.append(t1 - t0)

    return float(np.mean(tiempos)), float(np.std(tiempos))


# =============================================================================
# MEDICIÓN — TABTRANSFORMER
# =============================================================================

def cargar_tabtransformer(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_ckpt = ckpt["args"]

    n_features = len(ALL_FEATURE_COLS)
    n_classes  = len(ckpt["label_encoder_classes"])

    modelo = TabTransformerModel(
        n_features=n_features,
        n_classes=n_classes,
        d_model=args_ckpt["d_model"],
        nhead=args_ckpt["nhead"],
        num_layers=args_ckpt["num_layers"],
        dim_feedforward=args_ckpt["d_model"] * 4,
        dropout=args_ckpt["dropout"],
    ).to(device)
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()

    scaler_mean  = np.asarray(ckpt["scaler_mean"],  dtype=np.float32)
    scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)

    return modelo, scaler_mean, scaler_scale


def medir_tabtransformer(modelo, scaler_mean, scaler_scale,
                          X: pd.DataFrame, device, n_repeticiones: int):
    # Reordenar/seleccionar columnas exactamente como espera el modelo
    X_arr = X[ALL_FEATURE_COLS].values.astype(np.float32)
    X_scaled = (X_arr - scaler_mean) / scaler_scale
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        _ = modelo(X_tensor)  # warm-up

        tiempos = []
        for _ in range(n_repeticiones):
            t0 = time.perf_counter()
            _ = modelo(X_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            tiempos.append(t1 - t0)

    return float(np.mean(tiempos)), float(np.std(tiempos))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Medición de tiempo de inferencia — RF y TabTransformer"
    )
    parser.add_argument("--clean_csv", required=True, help="aa_clean.csv o ua_clean.csv")
    parser.add_argument("--ids_csv",   required=True, help="aa_ids.csv o ua_ids.csv")
    parser.add_argument("--clf_path",  required=True, help="clf_AA.joblib / clf_UA.joblib")
    parser.add_argument("--reg_path",  required=True, help="reg_AA.joblib / reg_UA.joblib")
    parser.add_argument("--tabt_ckpt", default=None,
                        help="tabtransformer_best.pt (solo para AA charge regression)")
    parser.add_argument("--label", default="AA", help="AA o UA (solo para nombrar filas)")
    parser.add_argument("--n_moleculas", type=int, default=5,
                        help="Cantidad de moléculas de distinto tamaño a probar")
    parser.add_argument("--n_repeticiones", type=int, default=20)
    parser.add_argument("--output", default="tiempos_inferencia.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    moleculas, feature_cols = cargar_moleculas(
        Path(args.clean_csv), Path(args.ids_csv), args.n_moleculas
    )
    print(f"Moléculas seleccionadas ({len(moleculas)}): "
          f"{[(n, t[0]) for n, t in moleculas.items()]}")

    clf = joblib.load(args.clf_path)
    reg = joblib.load(args.reg_path)

    tabt_modelo = None
    if args.tabt_ckpt:
        tabt_modelo, scaler_mean, scaler_scale = cargar_tabtransformer(
            Path(args.tabt_ckpt), device
        )

    resultados = []

    for nombre, (n_atomos, X) in moleculas.items():
        # RF — clasificación (atomtype)
        t_mean, t_std = medir_rf(clf, X, args.n_repeticiones)
        resultados.append({"modelo": f"RF_atomtype_{args.label}", "molecula": nombre,
                            "n_atomos": n_atomos, "tiempo_medio_s": t_mean,
                            "tiempo_std_s": t_std})
        print(f"[RF_atomtype_{args.label}] {nombre} (N={n_atomos}): "
              f"{t_mean*1000:.4f} ms ± {t_std*1000:.4f} ms")

        # RF — regresión (charge)
        t_mean, t_std = medir_rf(reg, X, args.n_repeticiones)
        resultados.append({"modelo": f"RF_charge_{args.label}", "molecula": nombre,
                            "n_atomos": n_atomos, "tiempo_medio_s": t_mean,
                            "tiempo_std_s": t_std})
        print(f"[RF_charge_{args.label}] {nombre} (N={n_atomos}): "
              f"{t_mean*1000:.4f} ms ± {t_std*1000:.4f} ms")

        # TabTransformer — solo si se pasó checkpoint (típicamente AA charge)
        if tabt_modelo is not None:
            t_mean, t_std = medir_tabtransformer(
                tabt_modelo, scaler_mean, scaler_scale, X, device,
                args.n_repeticiones
            )
            resultados.append({"modelo": "TabTransformer_charge_AA", "molecula": nombre,
                                "n_atomos": n_atomos, "tiempo_medio_s": t_mean,
                                "tiempo_std_s": t_std})
            print(f"[TabTransformer_charge_AA] {nombre} (N={n_atomos}): "
                  f"{t_mean*1000:.4f} ms ± {t_std*1000:.4f} ms")

    df_out = pd.DataFrame(resultados)
    df_out.to_csv(args.output, index=False)
    print(f"\n✔ Resultados guardados en: {args.output}")


if __name__ == "__main__":
    main()

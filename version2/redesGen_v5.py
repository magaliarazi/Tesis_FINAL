"""
tvae_synthesizer.py
====================
TVAE implementado con numpy puro — sin torch, sin sdv, sin scikit-learn.
Requiere solo: numpy y pandas.

Mejoras respecto a versiones anteriores:
  - Optimizador Adam por capa (momento 1 y 2) en lugar de SGD puro
  - Gradient clipping global en cada paso
  - lr_schedule: reduccion de lr si la loss no mejora (patience)
  - Arquitectura mas compacta por defecto (adecuada para ~500 filas)
  - log_var clampeado para evitar exp() explosivo
  - Deteccion automatica de tipos mejorada
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utilidades de activacion
# ---------------------------------------------------------------------------

def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0.0).astype(np.float32)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / (e.sum(axis=1, keepdims=True) + 1e-8)

def he_init(fan_in, fan_out, rng):
    return rng.standard_normal((fan_in, fan_out)).astype(np.float32) * np.sqrt(2.0 / fan_in)


# ---------------------------------------------------------------------------
# Capa densa con optimizador Adam interno
# ---------------------------------------------------------------------------

class DenseLayer:
    def __init__(self, in_dim, out_dim, activation="relu", rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        self.W    = he_init(in_dim, out_dim, rng)
        self.b    = np.zeros((1, out_dim), dtype=np.float32)
        self.act  = activation
        # Cache backprop
        self.x = self.z = None
        # Adam moments
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t  = 0

    def forward(self, x):
        self.x = x.astype(np.float32)
        self.z = self.x @ self.W + self.b
        return relu(self.z) if self.act == "relu" else self.z

    def backward(self, d_out, lr, l2, beta1=0.9, beta2=0.999, eps=1e-8):
        d_out = d_out.astype(np.float32)
        if self.act == "relu":
            d_out = d_out * relu_grad(self.z)
        n   = max(len(self.x), 1)
        dW  = self.x.T @ d_out / n + l2 * self.W
        db  = d_out.mean(axis=0, keepdims=True)
        dx  = d_out @ self.W.T
        # Adam update
        self.t  += 1
        self.mW  = beta1 * self.mW + (1 - beta1) * dW
        self.vW  = beta2 * self.vW + (1 - beta2) * dW**2
        self.mb  = beta1 * self.mb + (1 - beta1) * db
        self.vb  = beta2 * self.vb + (1 - beta2) * db**2
        mW_hat   = self.mW / (1 - beta1**self.t)
        vW_hat   = self.vW / (1 - beta2**self.t)
        mb_hat   = self.mb / (1 - beta1**self.t)
        vb_hat   = self.vb / (1 - beta2**self.t)
        self.W  -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.b  -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
        return dx


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder:
    def __init__(self, input_dim, hidden_dims, latent_dim, rng):
        dims = [input_dim] + list(hidden_dims)
        self.layers    = [DenseLayer(dims[i], dims[i+1], "relu", rng) for i in range(len(dims)-1)]
        self.mu_layer  = DenseLayer(dims[-1], latent_dim, "linear", rng)
        self.lv_layer  = DenseLayer(dims[-1], latent_dim, "linear", rng)

    def forward(self, x):
        h = x.astype(np.float32)
        for layer in self.layers:
            h = layer.forward(h)
        self.h   = h
        mu       = self.mu_layer.forward(h)
        log_var  = np.clip(self.lv_layer.forward(h), -4, 4)  # evita exp() explosivo
        return mu, log_var

    def backward(self, d_mu, d_lv, lr, l2):
        dh  = self.mu_layer.backward(d_mu, lr, l2)
        dh += self.lv_layer.backward(d_lv, lr, l2)
        for layer in reversed(self.layers):
            dh = layer.backward(dh, lr, l2)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder:
    def __init__(self, latent_dim, hidden_dims, output_dim, rng):
        dims = [latent_dim] + list(hidden_dims)
        self.layers    = [DenseLayer(dims[i], dims[i+1], "relu", rng) for i in range(len(dims)-1)]
        self.out_layer = DenseLayer(dims[-1], output_dim, "linear", rng)

    def forward(self, z):
        h = z.astype(np.float32)
        for layer in self.layers:
            h = layer.forward(h)
        return self.out_layer.forward(h)

    def backward(self, d_out, lr, l2):
        dh = self.out_layer.backward(d_out, lr, l2)
        for layer in reversed(self.layers):
            dh = layer.backward(dh, lr, l2)
        return dh


# ---------------------------------------------------------------------------
# Preprocesador
# ---------------------------------------------------------------------------

class DataTransformer:
    def __init__(self):
        self.col_info   = []
        self.output_dim = 0

    def fit(self, df, col_types):
        self.col_info   = []
        self.output_dim = 0
        for col in df.columns:
            sdtype = col_types.get(col, "numerical")
            series = df[col].dropna()

            if sdtype == "boolean":
                info = {"col": col, "type": "boolean",
                        "start": self.output_dim, "dim": 1}
                self.output_dim += 1

            elif sdtype == "categorical":
                cats    = sorted(series.astype(str).unique())
                cat2idx = {c: i for i, c in enumerate(cats)}
                info = {"col": col, "type": "categorical",
                        "start": self.output_dim, "dim": len(cats),
                        "cats": cats, "cat2idx": cat2idx}
                self.output_dim += len(cats)

            else:  # numerical
                mean    = float(series.mean())
                std     = float(series.std()) or 1.0
                min_val = float(series.min())
                max_val = float(series.max())
                info = {"col": col, "type": "numerical",
                        "start": self.output_dim, "dim": 1,
                        "mean": mean, "std": std,
                        "min_val": min_val, "max_val": max_val}
                self.output_dim += 1

            self.col_info.append(info)
        return self

    def transform(self, df):
        n   = len(df)
        out = np.zeros((n, self.output_dim), dtype=np.float32)
        for info in self.col_info:
            col = info["col"]
            s   = info["start"]
            series = df[col]

            if info["type"] == "boolean":
                vals = series.fillna(0).astype(float).values
                out[:, s] = np.where(vals > 0.5, 1.0, 0.0)

            elif info["type"] == "categorical":
                for i, val in enumerate(series.astype(str).values):
                    idx = info["cat2idx"].get(val, 0)
                    out[i, s + idx] = 1.0

            else:
                vals = series.fillna(info["mean"]).astype(float).values
                norm = (vals - info["mean"]) / (info["std"] * 4 + 1e-8)
                out[:, s] = np.clip(norm, -1, 1)

        return out

    def inverse_transform(self, data):
        result = {}
        for info in self.col_info:
            col   = info["col"]
            s     = info["start"]
            d     = info["dim"]
            chunk = data[:, s:s+d]

            if info["type"] == "boolean":
                vals = chunk[:, 0]
                threshold = float(np.median(vals))
                result[col] = (vals > threshold).astype(int)

            elif info["type"] == "categorical":
                indices = np.argmax(chunk, axis=1)
                indices = np.clip(indices, 0, len(info["cats"]) - 1)
                result[col] = [info["cats"][i] for i in indices]

            else:
                val = chunk[:, 0] * info["std"] * 4 + info["mean"]
                val = np.clip(val, info["min_val"], info["max_val"])
                result[col] = val.astype(np.float32)

        return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Perdida mixta + gradientes
# ---------------------------------------------------------------------------

def compute_loss_and_grads(recon, x, mu, log_var, col_info, loss_factor=2.0):
    n       = max(len(x), 1)
    d_recon = np.zeros_like(recon, dtype=np.float32)
    loss    = 0.0

    for info in col_info:
        s = info["start"]
        d = info["dim"]
        r = recon[:, s:s+d]
        t = x[:, s:s+d]

        if info["type"] == "boolean":
            p     = sigmoid(r)
            l     = -np.mean(t * np.log(p + 1e-8) + (1-t) * np.log(1-p + 1e-8))
            loss += l
            d_recon[:, s:s+d] = (p - t) / n

        elif info["type"] == "categorical":
            p     = softmax(r)
            l     = loss_factor * -np.mean(np.sum(t * np.log(p + 1e-8), axis=1))
            loss += l
            d_recon[:, s:s+d] = loss_factor * (p - t) / n

        else:
            diff  = r - t
            loss += np.mean(diff**2)
            d_recon[:, s:s+d] = 2 * diff / n

    # KL divergence
    kl    = -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))
    loss += kl

    d_mu  = mu / n
    d_lv  = 0.5 * (np.exp(np.clip(log_var, -4, 4)) - 1) / n

    return float(loss), d_recon, d_mu, d_lv


# ---------------------------------------------------------------------------
# Gradient clipping global
# ---------------------------------------------------------------------------

def clip_grad(g, clip):
    norm = np.linalg.norm(g)
    if norm > clip:
        g = g * (clip / (norm + 1e-8))
    return g


# ---------------------------------------------------------------------------
# Deteccion automatica de tipos
# ---------------------------------------------------------------------------

def _detect_column_types(df, categorical_max_unique=30):
    col_types = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            col_types[col] = "numerical"
            continue

        if series.dtype == object or pd.api.types.is_string_dtype(series):
            col_types[col] = "categorical"
            continue

        if pd.api.types.is_bool_dtype(series):
            col_types[col] = "boolean"
            continue

        unique_vals = set(series.unique())
        if unique_vals <= {0, 1, 0.0, 1.0}:
            col_types[col] = "boolean"
            continue

        if pd.api.types.is_integer_dtype(series) and series.nunique() <= categorical_max_unique:
            col_types[col] = "categorical"
            continue

        col_types[col] = "numerical"
    return col_types


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def _load_data(data, verbose):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Formato no soportado: '{suffix}'")
    if verbose:
        print(f"[Carga] {path.name} leido correctamente.")
    return df


# ---------------------------------------------------------------------------
# Comparacion de distribuciones
# ---------------------------------------------------------------------------

def compare_distributions(real, synthetic, cols=None, max_cols=10):
    """
    Tabla comparativa de estadisticas descriptivas (numericas).
    """
    numeric_cols = real.select_dtypes(include="number").columns.tolist()
    if cols:
        numeric_cols = [c for c in cols if c in numeric_cols]
    numeric_cols = numeric_cols[:max_cols]

    rows = []
    for col in numeric_cols:
        r = real[col].astype(float)
        s = synthetic[col].astype(float)
        rows.append({
            "columna":    col,
            "real_mean":  round(float(r.mean()),   4),
            "synth_mean": round(float(s.mean()),   4),
            "real_std":   round(float(r.std()),    4),
            "synth_std":  round(float(s.std()),    4),
            "real_min":   round(float(r.min()),    4),
            "synth_min":  round(float(s.min()),    4),
            "real_p50":   round(float(r.median()), 4),
            "synth_p50":  round(float(s.median()), 4),
            "real_max":   round(float(r.max()),    4),
            "synth_max":  round(float(s.max()),    4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------

def run_tvae(
    data,
    num_rows=500,
    epochs=500,
    batch_size=32,
    embedding_dim=32,
    compress_dims=(128, 64),
    decompress_dims=(64, 128),
    lr=1e-4,
    l2scale=1e-5,
    loss_factor=2.0,
    grad_clip=1.0,
    categorical_max_unique=30,
    patience=50,
    output_path=None,
    verbose=True,
    random_state=42,
):
    """
    Entrena un TVAE con numpy+Adam y genera datos sinteticos.

    Parametros
    ----------
    data                  : DataFrame, ruta CSV / TSV / Parquet / Excel.
    num_rows              : filas sinteticas a generar.
    epochs                : epocas maximas de entrenamiento.
    batch_size            : tamano de mini-lote (32 recomendado para ~500 filas).
    embedding_dim         : dimension del espacio latente.
    compress_dims         : neuronas por capa del codificador.
    decompress_dims       : neuronas por capa del decodificador.
    lr                    : tasa de aprendizaje Adam (1e-4 es seguro).
    l2scale               : regularizacion L2.
    loss_factor           : peso extra en perdida categorica.
    grad_clip             : clipping global de gradientes (evita explosion).
    categorical_max_unique: tope de unicos para columnas enteras -> categoricas.
    patience              : epocas sin mejora antes de reducir lr a la mitad.
    output_path           : ruta para guardar CSV sintetico (opcional).
    verbose               : imprimir progreso.
    random_state          : semilla aleatoria.

    Retorna
    -------
    dict con:
        'synthetic'    -> DataFrame sintetico
        'encoder'      -> Encoder entrenado
        'decoder'      -> Decoder entrenado
        'transformer'  -> DataTransformer ajustado
        'col_types'    -> dict de tipos por columna
        'losses'       -> lista de perdidas por epoca
    """
    rng = np.random.default_rng(random_state)
    np.random.seed(random_state)

    # 1. Carga
    df = _load_data(data, verbose)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
        print(f"{'='*55}")

    # 2. Limpieza
    df = df.dropna(how="all").reset_index(drop=True)

    # 3. Tipos
    col_types = _detect_column_types(df, categorical_max_unique)
    if verbose:
        counts = {}
        for t in col_types.values():
            counts[t] = counts.get(t, 0) + 1
        print(f"[Tipos] " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    # 4. Transformacion
    transformer = DataTransformer()
    transformer.fit(df, col_types)
    data_np = transformer.transform(df).astype(np.float32)

    if verbose:
        print(f"[Preprocesamiento] Vector: {data_np.shape[1]} dimensiones")

    input_dim = data_np.shape[1]

    # 5. Modelo
    encoder = Encoder(input_dim, compress_dims, embedding_dim, rng)
    decoder = Decoder(embedding_dim, decompress_dims, input_dim, rng)

    # 6. Entrenamiento
    if verbose:
        print(f"\n[TVAE] Entrenando - {epochs} epocas | lr={lr} | "
              f"batch={batch_size} | clip={grad_clip}")

    n         = len(data_np)
    losses    = []
    best_loss = np.inf
    no_improve = 0
    current_lr = lr
    log_every  = max(1, epochs // 10)

    for epoch in range(1, epochs + 1):
        idx        = rng.permutation(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            batch = data_np[idx[start:start + batch_size]].copy()

            # Forward encoder
            mu, log_var = encoder.forward(batch)

            # Reparametrizacion
            eps = rng.standard_normal(mu.shape).astype(np.float32)
            z   = mu + eps * np.exp(0.5 * log_var)

            # Forward decoder
            recon = decoder.forward(z)

            # Perdida y gradientes
            loss, d_recon, d_mu, d_lv = compute_loss_and_grads(
                recon, batch, mu, log_var, transformer.col_info, loss_factor
            )

            if not np.isfinite(loss):
                continue

            # Clip gradientes
            d_recon = clip_grad(d_recon, grad_clip)

            # Backprop decoder
            dz = decoder.backward(d_recon, current_lr, l2scale)

            # Gradientes encoder via reparametrizacion
            d_mu_total = clip_grad(d_mu + dz, grad_clip)
            d_lv_total = clip_grad(
                d_lv + dz * eps * 0.5 * np.exp(np.clip(0.5 * log_var, -4, 4)),
                grad_clip
            )

            # Backprop encoder
            encoder.backward(d_mu_total, d_lv_total, current_lr, l2scale)

            epoch_loss += loss
            n_batches  += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)

        # Learning rate schedule por patience
        if avg < best_loss - 1e-4:
            best_loss  = avg
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                current_lr *= 0.5
                no_improve  = 0
                if verbose:
                    print(f"  [lr schedule] lr reducido a {current_lr:.2e}")

        if verbose and (epoch % log_every == 0 or epoch == 1):
            print(f"  Epoca {epoch:>4}/{epochs}  |  loss={avg:.4f}  |  lr={current_lr:.2e}")

    if verbose:
        print("[TVAE] Entrenamiento completado.")

    # 7. Generacion
    if verbose:
        print(f"\n[TVAE] Generando {num_rows} filas sinteticas ...")

    z_new     = rng.standard_normal((num_rows, embedding_dim)).astype(np.float32)
    recon_new = decoder.forward(z_new)
    synthetic = transformer.inverse_transform(recon_new)

    if verbose:
        print(f"[TVAE] Generacion exitosa: {synthetic.shape}")

    # 8. Guardar
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(out, index=False)
        if verbose:
            print(f"\n[Output] CSV guardado en: {out.resolve()}")

    return {
        "synthetic":   synthetic,
        "encoder":     encoder,
        "decoder":     decoder,
        "transformer": transformer,
        "col_types":   col_types,
        "losses":      losses,
    }


def sample_from_trained(decoder, transformer, num_rows=500,
                        embedding_dim=None, output_path=None, random_state=42):
    """
    Genera nuevas muestras desde un decoder ya entrenado sin re-entrenar.

    Parametros
    ----------
    decoder      : Decoder entrenado (result['decoder']).
    transformer  : DataTransformer ajustado (result['transformer']).
    num_rows     : filas a generar.
    embedding_dim: dimension del espacio latente (se infiere si no se pasa).
    output_path  : ruta opcional para guardar CSV.
    random_state : semilla.
    """
    if embedding_dim is None:
        embedding_dim = decoder.layers[0].W.shape[0]
    rng      = np.random.default_rng(random_state)
    z        = rng.standard_normal((num_rows, embedding_dim)).astype(np.float32)
    recon    = decoder.forward(z)
    synthetic = transformer.inverse_transform(recon)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(output_path, index=False)
        print(f"[Output] Guardado en: {Path(output_path).resolve()}")
    return synthetic


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TVAE - Generador de datos sinteticos tabulares (numpy puro)"
    )
    parser.add_argument("--input",          required=True,
                        help="Ruta al dataset de entrada (CSV, TSV, Parquet, Excel)")
    parser.add_argument("--output_dir",     default=".",
                        help="Directorio donde guardar los resultados (default: .)")
    parser.add_argument("--num_rows",       type=int,   default=500,
                        help="Filas sinteticas a generar (default: 500)")
    parser.add_argument("--epochs",         type=int,   default=500,
                        help="Epocas de entrenamiento (default: 500)")
    parser.add_argument("--batch_size",     type=int,   default=32,
                        help="Tamano de mini-lote (default: 32)")
    parser.add_argument("--embedding_dim",  type=int,   default=32,
                        help="Dimension del espacio latente (default: 32)")
    parser.add_argument("--lr",             type=float, default=1e-4,
                        help="Tasa de aprendizaje Adam (default: 0.0001)")
    parser.add_argument("--grad_clip",      type=float, default=1.0,
                        help="Clipping de gradientes (default: 1.0)")
    parser.add_argument("--loss_factor",    type=float, default=2.0,
                        help="Peso en perdida categorica (default: 2.0)")
    parser.add_argument("--patience",       type=int,   default=50,
                        help="Epocas sin mejora para reducir lr (default: 50)")
    parser.add_argument("--cat_max_unique", type=int,   default=30,
                        help="Tope de unicos para columnas enteras->categoricas (default: 30)")
    parser.add_argument("--random_state",   type=int,   default=42,
                        help="Semilla aleatoria (default: 42)")
    parser.add_argument("--no_compare",     action="store_true",
                        help="Omitir tabla de comparacion de distribuciones")

    args = parser.parse_args()

    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_stem  = Path(args.input).stem
    output_csv  = out_dir / f"{input_stem}_synthetic.csv"
    compare_csv = out_dir / f"{input_stem}_distribution_comparison.csv"

    result = run_tvae(
        data             = args.input,
        num_rows         = args.num_rows,
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        embedding_dim    = args.embedding_dim,
        lr               = args.lr,
        grad_clip        = args.grad_clip,
        loss_factor      = args.loss_factor,
        patience         = args.patience,
        categorical_max_unique = args.cat_max_unique,
        output_path      = output_csv,
        random_state     = args.random_state,
        verbose          = True,
    )

    print(f"\n[Output] Datos sinteticos guardados en: {output_csv.resolve()}")
    print("\n[Muestra] Primeras 5 filas:")
    print(result["synthetic"].head().to_string(index=False))

    if not args.no_compare:
        real_df = _load_data(args.input, verbose=False)
        comp    = compare_distributions(real_df, result["synthetic"])
        comp.to_csv(compare_csv, index=False)
        print(f"\n[Output] Comparacion guardada en: {compare_csv.resolve()}")
        print("\n[Comparacion de distribuciones]")
        print(comp.to_string(index=False))

    print(f"\n{'='*55}")
    print("  Archivos generados:")
    print(f"  - {output_csv.resolve()}")
    if not args.no_compare:
        print(f"  - {compare_csv.resolve()}")
    print(f"{'='*55}")

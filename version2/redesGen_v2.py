"""
tvae_synthesizer.py
====================
TVAE implementado desde cero — sin dependencia de SDV.
Requiere solo: pip install torch pandas numpy scikit-learn

Soporta variables numericas, binarias y categoricas (mixed-type tabular data).
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Preprocesador: VGM para numericas + one-hot para categoricas/binarias
# ---------------------------------------------------------------------------

class DataTransformer:
    """
    Transforma un DataFrame mixto en un tensor continuo para el TVAE:
      - Numericas   : modo gaussiano via BayesianGaussianMixture (VGM)
      - Booleanas   : se tratan como binarias {0, 1}
      - Categoricas : one-hot encoding
    Guarda toda la info necesaria para invertir la transformacion.
    """

    def __init__(self, n_components=5):
        self.n_components = n_components
        self.col_info = []      # metadata por columna
        self.output_dim = 0     # dimension total del vector transformado

    def fit(self, df: pd.DataFrame, col_types: dict):
        self.col_info = []
        self.output_dim = 0

        for col in df.columns:
            sdtype = col_types.get(col, "numerical")
            series = df[col].fillna(df[col].mode()[0] if sdtype == "categorical" else df[col].median())

            if sdtype == "boolean":
                info = {"col": col, "type": "boolean", "start": self.output_dim, "dim": 1}
                self.output_dim += 1

            elif sdtype == "categorical":
                le = LabelEncoder()
                le.fit(series.astype(str))
                n_cats = len(le.classes_)
                info = {"col": col, "type": "categorical", "start": self.output_dim,
                        "dim": n_cats, "le": le}
                self.output_dim += n_cats

            else:  # numerical — VGM
                vgm = BayesianGaussianMixture(
                    n_components=self.n_components,
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    random_state=42,
                )
                vgm.fit(series.values.reshape(-1, 1))
                # dim = 1 (valor normalizado) + n_components (modo one-hot)
                dim = 1 + self.n_components
                info = {"col": col, "type": "numerical", "start": self.output_dim,
                        "dim": dim, "vgm": vgm, "n_comp": self.n_components}
                self.output_dim += dim

            self.col_info.append(info)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for info in self.col_info:
            col = info["col"]
            series = df[col].fillna(0)

            if info["type"] == "boolean":
                parts.append(series.values.reshape(-1, 1).astype(np.float32))

            elif info["type"] == "categorical":
                le = info["le"]
                encoded = le.transform(series.astype(str).values)
                onehot = np.zeros((len(df), info["dim"]), dtype=np.float32)
                onehot[np.arange(len(df)), encoded] = 1.0
                parts.append(onehot)

            else:  # numerical
                vgm = info["vgm"]
                vals = series.values.reshape(-1, 1)
                means = vgm.means_.flatten()
                stds  = np.sqrt(vgm.covariances_).flatten()
                # Componente mas probable para cada muestra
                modes = vgm.predict(vals)
                # Valor normalizado segun la componente asignada
                normalized = (vals.flatten() - means[modes]) / (4 * stds[modes] + 1e-8)
                normalized = np.clip(normalized, -1, 1).astype(np.float32)
                # One-hot del modo
                mode_onehot = np.zeros((len(df), info["n_comp"]), dtype=np.float32)
                mode_onehot[np.arange(len(df)), modes] = 1.0
                parts.append(normalized.reshape(-1, 1))
                parts.append(mode_onehot)

        return np.concatenate(parts, axis=1)

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        result = {}
        for info in self.col_info:
            col   = info["col"]
            start = info["start"]
            dim   = info["dim"]
            chunk = data[:, start:start+dim]

            if info["type"] == "boolean":
                result[col] = (chunk[:, 0] > 0.5).astype(int)

            elif info["type"] == "categorical":
                indices = np.argmax(chunk, axis=1)
                indices = np.clip(indices, 0, len(info["le"].classes_) - 1)
                result[col] = info["le"].inverse_transform(indices)

            else:  # numerical
                norm_val = chunk[:, 0]
                mode_onehot = chunk[:, 1:]
                modes = np.argmax(mode_onehot, axis=1)
                vgm   = info["vgm"]
                means = vgm.means_.flatten()
                stds  = np.sqrt(vgm.covariances_).flatten()
                reconstructed = norm_val * 4 * stds[modes] + means[modes]
                result[col] = reconstructed.astype(np.float32)

        return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Arquitectura TVAE
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim, compress_dims, embedding_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for d in compress_dims:
            layers += [nn.Linear(prev, d), nn.ReLU()]
            prev = d
        self.net  = nn.Sequential(*layers)
        self.mu   = nn.Linear(prev, embedding_dim)
        self.logvar = nn.Linear(prev, embedding_dim)

    def forward(self, x):
        h      = self.net(x)
        mu     = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, embedding_dim, decompress_dims, output_dim):
        super().__init__()
        layers = []
        prev = embedding_dim
        for d in decompress_dims:
            layers += [nn.Linear(prev, d), nn.ReLU()]
            prev = d
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class TVAE(nn.Module):
    def __init__(self, input_dim, compress_dims, embedding_dim, decompress_dims):
        super().__init__()
        self.encoder = Encoder(input_dim, compress_dims, embedding_dim)
        self.decoder = Decoder(embedding_dim, decompress_dims, input_dim)
        self.embedding_dim = embedding_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar


# ---------------------------------------------------------------------------
# Perdida TVAE: reconstruccion mixta + KL divergence
# ---------------------------------------------------------------------------

def tvae_loss(recon, x, mu, logvar, col_info, loss_factor=2.0):
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    recon_loss = 0.0

    for info in col_info:
        start = info["start"]
        dim   = info["dim"]
        r = recon[:, start:start+dim]
        t = x[:, start:start+dim]

        if info["type"] == "boolean":
            recon_loss += nn.functional.binary_cross_entropy_with_logits(r, t)

        elif info["type"] == "categorical":
            target_idx = t.argmax(dim=1)
            recon_loss += loss_factor * nn.functional.cross_entropy(r, target_idx)

        else:  # numerical: MSE para valor + cross_entropy para modo
            recon_loss += nn.functional.mse_loss(r[:, 0], t[:, 0])
            target_mode = t[:, 1:].argmax(dim=1)
            recon_loss += loss_factor * nn.functional.cross_entropy(r[:, 1:], target_mode)

    return recon_loss + kl


# ---------------------------------------------------------------------------
# Deteccion automatica de tipos
# ---------------------------------------------------------------------------

def _detect_column_types(df: pd.DataFrame, categorical_max_unique: int = 30) -> dict:
    col_types = {}
    for col in df.columns:
        series = df[col].dropna()

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

def _load_data(data, verbose: bool) -> pd.DataFrame:
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
# Evaluacion simple sin SDV
# ---------------------------------------------------------------------------

def compare_distributions(real, synthetic, cols=None, max_cols=10):
    """
    Compara media, std, min, p50 y max entre datos reales y sinteticos
    para columnas numericas.
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
    epochs=300,
    batch_size=64,
    embedding_dim=128,
    compress_dims=(256, 128),
    decompress_dims=(128, 256),
    l2scale=1e-5,
    loss_factor=2.0,
    categorical_max_unique=30,
    n_components=5,
    output_path=None,
    verbose=True,
) -> dict:
    """
    Entrena un TVAE desde cero y genera datos sinteticos.

    Parametros
    ----------
    data                  : DataFrame, ruta CSV / TSV / Parquet / Excel.
    num_rows              : filas sinteticas a generar.
    epochs                : epocas de entrenamiento.
    batch_size            : tamano de mini-lote.
    embedding_dim         : dimension del espacio latente.
    compress_dims         : neuronas por capa del codificador.
    decompress_dims       : neuronas por capa del decodificador.
    l2scale               : regularizacion L2 del optimizador.
    loss_factor           : peso extra en perdida categorica.
    categorical_max_unique: tope unicos para columnas enteras -> categoricas.
    n_components          : componentes del VGM para columnas numericas.
    output_path           : ruta para guardar CSV sintetico (opcional).
    verbose               : imprimir progreso.

    Retorna
    -------
    dict con:
        'synthetic'    -> DataFrame sintetico
        'model'        -> TVAE entrenado (nn.Module)
        'transformer'  -> DataTransformer ajustado
        'col_types'    -> dict de tipos por columna
        'losses'       -> lista de perdidas por epoca
    """
    # 1. Carga
    df = _load_data(data, verbose)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
        print(f"{'='*55}")

    # 2. Limpieza minima
    df = df.dropna(how="all").reset_index(drop=True)

    # 3. Tipos de columna
    col_types = _detect_column_types(df, categorical_max_unique)

    if verbose:
        counts = {"numerical": 0, "boolean": 0, "categorical": 0}
        for t in col_types.values():
            counts[t] = counts.get(t, 0) + 1
        print(f"[Tipos] numerical={counts['numerical']}  "
              f"boolean={counts['boolean']}  "
              f"categorical={counts['categorical']}")

    # 4. Transformacion
    transformer = DataTransformer(n_components=n_components)
    transformer.fit(df, col_types)
    data_np = transformer.transform(df)

    if verbose:
        print(f"[Preprocesamiento] Vector transformado: {data_np.shape[1]} dimensiones")

    # 5. DataLoader
    tensor_data = torch.tensor(data_np, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(tensor_data),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # 6. Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[Dispositivo] {device}")

    model = TVAE(
        input_dim=data_np.shape[1],
        compress_dims=compress_dims,
        embedding_dim=embedding_dim,
        decompress_dims=decompress_dims,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=l2scale)

    # 7. Entrenamiento
    if verbose:
        print(f"\n[TVAE] Entrenando — {epochs} epocas ...")

    losses = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = tvae_loss(recon, batch, mu, logvar,
                             transformer.col_info, loss_factor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            print(f"  Epoca {epoch:>4}/{epochs}  |  loss={avg_loss:.4f}")

    if verbose:
        print("[TVAE] Entrenamiento completado.")

    # 8. Generacion
    if verbose:
        print(f"\n[TVAE] Generando {num_rows} filas sinteticas ...")

    model.eval()
    with torch.no_grad():
        z = torch.randn(num_rows, embedding_dim).to(device)
        recon_np = model.decoder(z).cpu().numpy()

    synthetic = transformer.inverse_transform(recon_np)

    if verbose:
        print(f"[TVAE] Generacion exitosa: {synthetic.shape}")

    # 9. Guardar
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(out, index=False)
        if verbose:
            print(f"\n[Output] CSV guardado en: {out.resolve()}")

    return {
        "synthetic":   synthetic,
        "model":       model,
        "transformer": transformer,
        "col_types":   col_types,
        "losses":      losses,
    }


def sample_from_trained(model, transformer, num_rows=500, output_path=None) -> pd.DataFrame:
    """
    Genera nuevas muestras desde un modelo ya entrenado sin re-entrenar.

    Parametros
    ----------
    model       : TVAE entrenado (result['model']).
    transformer : DataTransformer ajustado (result['transformer']).
    num_rows    : cantidad de filas a generar.
    output_path : ruta opcional para guardar CSV.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_rows, model.embedding_dim).to(device)
        recon_np = model.decoder(z).cpu().numpy()
    synthetic = transformer.inverse_transform(recon_np)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(output_path, index=False)
        print(f"[Output] Guardado en: {Path(output_path).resolve()}")
    return synthetic


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

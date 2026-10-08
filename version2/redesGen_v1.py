from __future__ import annotations
 
import warnings
from pathlib import Path
from typing import Literal
 
import numpy as np
import pandas as pd
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
 
warnings.filterwarnings("ignore")
 
 
# ---------------------------------------------------------------------------
# Detección automática de tipos de columna
# ---------------------------------------------------------------------------
 
def _detect_column_types(
    df: pd.DataFrame,
    boolean_threshold: float = 0.05,
    categorical_max_unique: int = 30,
) -> dict[str, str]:
    """
    Infiere el sdtype de cada columna:
      - 'boolean'     : columnas numéricas con solo valores 0/1 o True/False
      - 'categorical' : object/string O numéricas con pocos valores únicos
      - 'numerical'   : resto de columnas numéricas
 
    Parámetros
    ----------
    df : DataFrame de entrada.
    boolean_threshold : fracción máxima de valores distintos a {0,1} para
                        considerar la columna booleana.
    categorical_max_unique : tope de categorías únicas para columnas numéricas
                             que en realidad son categóricas (ej. atomtype codes).
    """
    col_types: dict[str, str] = {}
 
    for col in df.columns:
        series = df[col].dropna()
 
        # Columnas object/string → categóricas
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            col_types[col] = "categorical"
            continue
 
        # Booleanas nativas de pandas
        if pd.api.types.is_bool_dtype(series):
            col_types[col] = "boolean"
            continue
 
        # Numéricas: revisar si son binarias {0, 1}
        unique_vals = set(series.unique())
        if unique_vals <= {0, 1, 0.0, 1.0, True, False}:
            col_types[col] = "boolean"
            continue
 
        # Numéricas enteras con pocos valores únicos → categóricas
        if (
            pd.api.types.is_integer_dtype(series)
            and series.nunique() <= categorical_max_unique
        ):
            col_types[col] = "categorical"
            continue
 
        # Por defecto: numérica
        col_types[col] = "numerical"
 
    return col_types
 
 
# ---------------------------------------------------------------------------
# Construcción de metadata SDV
# ---------------------------------------------------------------------------
 
def _build_metadata(
    df: pd.DataFrame,
    col_types: dict[str, str],
    primary_key: str | None = None,
) -> SingleTableMetadata:
    """
    Crea y configura un objeto SingleTableMetadata a partir del
    diccionario de tipos inferido.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
 
    for col, sdtype in col_types.items():
        metadata.update_column(col, sdtype=sdtype)
 
    if primary_key:
        metadata.set_primary_key(primary_key)
    elif metadata.primary_key:
        # Si SDV detectó una PK automática, la quitamos para no
        # restringir la generación
        metadata.primary_key = None
 
    return metadata
 
 
# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
 
def run_tvae(
    data: pd.DataFrame | str | Path,
    *,
    num_rows: int = 500,
    epochs: int = 300,
    batch_size: int = 64,
    embedding_dim: int = 128,
    compress_dims: tuple[int, ...] = (256, 128),
    decompress_dims: tuple[int, ...] = (128, 256),
    l2scale: float = 1e-5,
    loss_factor: float = 2.0,
    primary_key: str | None = None,
    boolean_threshold: float = 0.05,
    categorical_max_unique: int = 30,
    evaluate: bool = True,
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    Entrena un TVAE sobre `data` y genera `num_rows` filas sintéticas.
 
    Parámetros
    ----------
    data : DataFrame, ruta CSV, ruta Parquet, o ruta Excel con los datos reales.
    num_rows : cantidad de filas sintéticas a generar.
    epochs : épocas de entrenamiento del TVAE.
    batch_size : tamaño de mini-lote.
    embedding_dim : dimensión del espacio latente (z).
    compress_dims : neuronas por capa en el codificador.
    decompress_dims : neuronas por capa en el decodificador.
    l2scale : regularización L2.
    loss_factor : peso adicional en la pérdida de reconstrucción
                  para columnas categóricas.
    primary_key : nombre de la columna clave primaria (opcional).
    boolean_threshold : umbral para detección automática de booleanas.
    categorical_max_unique : máx. únicos para columnas enteras → categóricas.
    evaluate : si True, calcula métricas de calidad SDV al final.
    output_path : ruta donde guardar el CSV sintético (opcional).
    verbose : imprime progreso y resumen.
 
    Retorna
    -------
    dict con claves:
        'synthetic'    : DataFrame con datos sintéticos.
        'synthesizer'  : objeto TVAESynthesizer entrenado.
        'metadata'     : objeto SingleTableMetadata.
        'col_types'    : dict de tipos detectados por columna.
        'quality'      : reporte de calidad SDV (si evaluate=True).
        'diagnostic'   : reporte de diagnóstico SDV (si evaluate=True).
    """
 
    # ------------------------------------------------------------------
    # 1. Carga de datos
    # ------------------------------------------------------------------
    df = _load_data(data, verbose)
 
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
        print(f"{'='*60}")
 
    # ------------------------------------------------------------------
    # 2. Limpieza básica
    # ------------------------------------------------------------------
    df = _basic_clean(df, verbose)
 
    # ------------------------------------------------------------------
    # 3. Detección de tipos de columna
    # ------------------------------------------------------------------
    col_types = _detect_column_types(df, boolean_threshold, categorical_max_unique)
 
    if verbose:
        _print_type_summary(col_types)
 
    # ------------------------------------------------------------------
    # 4. Metadata SDV
    # ------------------------------------------------------------------
    metadata = _build_metadata(df, col_types, primary_key)
 
    # ------------------------------------------------------------------
    # 5. Crear y entrenar el sintetizador
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[TVAE] Iniciando entrenamiento — {epochs} épocas, "
              f"embedding_dim={embedding_dim} ...")
 
    synthesizer = TVAESynthesizer(
        metadata,
        epochs=epochs,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        compress_dims=compress_dims,
        decompress_dims=decompress_dims,
        l2scale=l2scale,
        loss_factor=loss_factor,
        verbose=verbose,
    )
    synthesizer.fit(df)
 
    if verbose:
        print("[TVAE] Entrenamiento completado.")
 
    # ------------------------------------------------------------------
    # 6. Generación de datos sintéticos
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[TVAE] Generando {num_rows} filas sintéticas ...")
 
    synthetic = synthesizer.sample(num_rows=num_rows)
 
    if verbose:
        print(f"[TVAE] Generación exitosa: {synthetic.shape}")
 
    # ------------------------------------------------------------------
    # 7. Evaluación de calidad (opcional)
    # ------------------------------------------------------------------
    quality_report = None
    diagnostic_report = None
 
    if evaluate:
        if verbose:
            print("\n[Evaluación] Calculando métricas de calidad SDV ...")
 
        diagnostic_report = run_diagnostic(
            real_data=df,
            synthetic_data=synthetic,
            metadata=metadata,
            verbose=False,
        )
        quality_report = evaluate_quality(
            real_data=df,
            synthetic_data=synthetic,
            metadata=metadata,
            verbose=False,
        )
 
        if verbose:
            _print_quality_summary(quality_report, diagnostic_report)
 
    # ------------------------------------------------------------------
    # 8. Guardar resultado (opcional)
    # ------------------------------------------------------------------
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(out, index=False)
        if verbose:
            print(f"\n[Output] CSV guardado en: {out.resolve()}")
 
    return {
        "synthetic": synthetic,
        "synthesizer": synthesizer,
        "metadata": metadata,
        "col_types": col_types,
        "quality": quality_report,
        "diagnostic": diagnostic_report,
    }
 
 
# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------
 
def _load_data(data: pd.DataFrame | str | Path, verbose: bool) -> pd.DataFrame:
    """Acepta DataFrame, CSV, Parquet o Excel."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
 
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
 
    suffix = path.suffix.lower()
    loaders = {
        ".csv": pd.read_csv,
        ".tsv": lambda p: pd.read_csv(p, sep="\t"),
        ".parquet": pd.read_parquet,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }
    if suffix not in loaders:
        raise ValueError(
            f"Formato no soportado: '{suffix}'. "
            f"Usar: {list(loaders.keys())}"
        )
 
    if verbose:
        print(f"[Carga] Leyendo {path.name} ...")
    return loaders[suffix](path)
 
 
def _basic_clean(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """
    Limpieza mínima no destructiva:
    - Elimina columnas 100 % nulas.
    - Elimina filas 100 % nulas.
    - Avisa si hay NaN parciales (no los elimina para no perder info).
    """
    before = df.shape
 
    # Columnas completamente vacías
    all_null_cols = df.columns[df.isnull().all()].tolist()
    if all_null_cols:
        df = df.drop(columns=all_null_cols)
        if verbose:
            print(f"[Limpieza] Columnas 100% nulas eliminadas: {all_null_cols}")
 
    # Filas completamente vacías
    df = df.dropna(how="all")
 
    if verbose and df.shape != before:
        print(f"[Limpieza] Shape: {before} → {df.shape}")
 
    # Advertencia de NaN parciales
    partial_nulls = df.isnull().sum()
    partial_nulls = partial_nulls[partial_nulls > 0]
    if not partial_nulls.empty and verbose:
        print(
            f"[Limpieza] Columnas con NaN parciales "
            f"(TVAE los maneja internamente):\n"
            + "\n".join(
                f"  {col}: {n} NaN ({n/len(df):.1%})"
                for col, n in partial_nulls.items()
            )
        )
 
    return df.reset_index(drop=True)
 
 
def _print_type_summary(col_types: dict[str, str]) -> None:
    counts: dict[str, list[str]] = {"numerical": [], "boolean": [], "categorical": []}
    for col, t in col_types.items():
        counts.get(t, counts["numerical"]).append(col)
 
    print("\n[Tipos detectados]")
    for sdtype, cols in counts.items():
        if cols:
            preview = ", ".join(cols[:5])
            suffix = f" ... (+{len(cols)-5})" if len(cols) > 5 else ""
            print(f"  {sdtype:<12} ({len(cols):>3}): {preview}{suffix}")
 
 
def _print_quality_summary(quality, diagnostic) -> None:
    print("\n[Calidad SDV]")
    try:
        score = quality.get_score()
        print(f"  Score global          : {score:.3f} / 1.000")
    except Exception:
        pass
 
    try:
        props = quality.get_properties()
        for _, row in props.iterrows():
            print(f"  {row['Property']:<28}: {row['Score']:.3f}")
    except Exception:
        pass
 
    print("\n[Diagnóstico SDV]")
    try:
        diag = diagnostic.get_results()
        for _, row in diag.iterrows():
            status = "✓" if row.get("Result", "") == "Passed" else "✗"
            print(f"  {status} {row.get('Property', '')} — {row.get('Result', '')}")
    except Exception:
        pass
 
 
# ---------------------------------------------------------------------------
# Utilidades adicionales
# ---------------------------------------------------------------------------
 
def sample_from_trained(
    synthesizer: TVAESynthesizer,
    num_rows: int = 500,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Genera nuevas muestras desde un sintetizador ya entrenado
    sin volver a entrenar.
 
    Parámetros
    ----------
    synthesizer : TVAESynthesizer previamente entrenado con run_tvae().
    num_rows : cantidad de filas a generar.
    output_path : ruta opcional para guardar CSV.
 
    Retorna
    -------
    DataFrame con datos sintéticos.
    """
    synthetic = synthesizer.sample(num_rows=num_rows)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(output_path, index=False)
        print(f"[Output] Guardado en: {Path(output_path).resolve()}")
    return synthetic
 
 
def compare_distributions(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    cols: list[str] | None = None,
    max_cols: int = 10,
) -> pd.DataFrame:
    """
    Compara estadísticas descriptivas entre datos reales y sintéticos
    para columnas numéricas.
 
    Retorna un DataFrame con media, std, min, 50%, max para cada conjunto.
    """
    numeric_cols = real.select_dtypes(include="number").columns.tolist()
    if cols:
        numeric_cols = [c for c in cols if c in numeric_cols]
    numeric_cols = numeric_cols[:max_cols]
 
    rows = []
    for col in numeric_cols:
        r = real[col].describe()
        s = synthetic[col].describe()
        rows.append({
            "columna": col,
            "real_mean": round(r["mean"], 4),
            "synth_mean": round(s["mean"], 4),
            "real_std": round(r["std"], 4),
            "synth_std": round(s["std"], 4),
            "real_min": round(r["min"], 4),
            "synth_min": round(s["min"], 4),
            "real_p50": round(r["50%"], 4),
            "synth_p50": round(s["50%"], 4),
            "real_max": round(r["max"], 4),
            "synth_max": round(s["max"], 4),
        })
    return pd.DataFrame(rows)
 
 
# ---------------------------------------------------------------------------
# Ejemplo de uso directo
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
	main()

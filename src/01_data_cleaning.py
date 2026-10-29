"""
01_data_cleaning.py
Limpieza e ingesta del dataset "Productivity Prediction of Garment Employees" (UCI).

Pasos:
  1. Carga con ruta relativa (portable entre laptops).
  2. Corrige errores de captura en 'department' ("sweing" -> "sewing", espacios extra).
  3. Imputa WIP = 0 solo en Finishing (no hay producto en proceso en esa estación).
  4. Crea la variable objetivo 'missed_target' (1 = el equipo NO alcanzó su meta).
  5. Valida la calidad con aserciones y guarda data/processed/garments_clean.csv.

Uso:  python src/01_data_cleaning.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "garments_worker_productivity.csv"
OUT = ROOT / "data" / "processed" / "garments_clean.csv"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Texto: quitar espacios y corregir el typo de captura
    df["department"] = df["department"].str.strip().replace({"sweing": "sewing"})
    df["quarter"] = df["quarter"].str.strip()
    df["day"] = df["day"].str.strip()

    # 2) Fecha -> tipo fecha (se usa solo como referencia, no como predictor)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")

    # 3) Imputación con regla de negocio (no con la mediana):
    #    los nulos de WIP coinciden exactamente con los registros de Finishing
    #    -> en Finishing el producto ya está ensamblado, el WIP real es 0.
    nulls_by_dept = df.loc[df["wip"].isna(), "department"].value_counts()
    print("Nulos de WIP por departamento:\n", nulls_by_dept.to_string(), "\n")
    df.loc[df["department"] == "finishing", "wip"] = df.loc[
        df["department"] == "finishing", "wip"
    ].fillna(0)

    # 4) Variable objetivo (clasificación binaria)
    df["missed_target"] = (
        df["actual_productivity"] < df["targeted_productivity"]
    ).astype(int)

    return df


def validate(df: pd.DataFrame) -> None:
    """Aserciones de calidad (Sesión 4: Data Integrity & Quality)."""
    assert df.isna().sum().sum() == 0, "Quedan valores nulos"
    assert set(df["department"]) == {"sewing", "finishing"}, "Departamentos inesperados"
    assert df["team"].between(1, 12).all(), "Equipo fuera de rango 1-12"
    assert df["targeted_productivity"].between(0, 1).all(), "Meta fuera de [0, 1]"
    assert (df[["smv", "wip", "over_time", "incentive", "no_of_workers"]] >= 0).all().all(), \
        "Hay valores negativos imposibles"
    print("Aserciones de calidad: OK")


if __name__ == "__main__":
    raw = pd.read_csv(RAW)
    print(f"Dataset crudo: {raw.shape[0]} filas x {raw.shape[1]} columnas")
    print("Departamentos crudos:", raw["department"].unique().tolist(), "\n")

    clean_df = clean(raw)
    validate(clean_df)

    rate = clean_df["missed_target"].mean()
    print(f"\nRegistros que NO alcanzan la meta: {clean_df['missed_target'].sum()} "
          f"({rate:.1%})")
    print("Departamentos limpios:", clean_df["department"].unique().tolist())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUT, index=False)
    print(f"\nGuardado: {OUT.relative_to(ROOT)}  ->  {clean_df.shape}")

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.encode.encoder1d import Encoder1D
from src.encode.encoder2d import Encoder2D
from src.math.functions import Functions

IN_PATH = Path("./data/dataset_pre_preprocessing.csv")
# IN_PATH = Path("./data/final_validation.csv")
OUT_PATH = Path("./data/preprocessed.csv")
# OUT_PATH = Path("./data/final_validation_preprocessed.csv")

FIELD_2D_SPECS: Sequence[tuple[str, str, str]] = (
    ("mu_tipo", "mu_intensita", "mu"),
    ("b_tipo_1", "b_intensita_1", "b1"),
    ("b_tipo_2", "b_Intensita_2", "b2"),
    ("sigma_tipo", "sigma_intensita", "sigma"),
    ("forzante_tipo", "forzante_intensita", "f"),
)

FIELD_1D_SPECS: Sequence[tuple[str, str, str, int]] = (
    ("L1_bc_fun", "L1_bc_intensita", "bc1", 1),
    ("L2_bc_fun", "L2_bc_intensita", "bc2", 2),
    ("L3_bc_fun", "L3_bc_intensita", "bc3", 3),
    ("L4_bc_fun", "L4_bc_intensita", "bc4", 4),
)

def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def save_dataset(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False)

def expand_encoding(values: Iterable[float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{i}": value for i, value in enumerate(values)}

def build_pde_params(row: pd.Series) -> dict[str, float]:
    return {
        "mu": row["mu_intensita"],
        "b1": row["b_intensita_1"],
        "b2": row["b_Intensita_2"],
        "sigma": row["sigma_intensita"],
        "alpha": row["alpha"],
    }

def encode_row(
    row: pd.Series,
    math_helper: Functions,
    encoder_1d: Encoder1D,
    encoder_2d: Encoder2D,
) -> dict[str, float]:
    math_helper.update_params(build_pde_params(row))

    encoded_row: dict[str, float] = {
        "tol": row["toll"],
        "err": row["err_fin"],
    }

    for name_col, magnitude_col, prefix in FIELD_2D_SPECS:
        func = math_helper.get_by_name_2d(row[name_col], row[magnitude_col])
        encoded_row.update(expand_encoding(encoder_2d.get_encoding(func), prefix))

    for name_col, magnitude_col, prefix, border in FIELD_1D_SPECS:
        func = math_helper.get_by_name_1d(row[name_col], row[magnitude_col], border=border)
        encoded_row.update(expand_encoding(encoder_1d.get_encoding(func), prefix))

    return encoded_row

def process(
    data: pd.DataFrame,
    math_helper: Functions,
    encoder_1d: Encoder1D,
    encoder_2d: Encoder2D,
) -> pd.DataFrame:
    encoded_rows = [
        encode_row(row, math_helper, encoder_1d, encoder_2d)
        for _, row in data.iterrows()
    ]

    return pd.DataFrame(encoded_rows)

def main() -> None:
    math_helper = Functions()
    encoder_1d = Encoder1D()
    encoder_2d = Encoder2D()

    raw = load_dataset(IN_PATH)
    encoded = process(raw, math_helper, encoder_1d, encoder_2d)
    save_dataset(encoded, OUT_PATH)

if __name__ == "__main__":
    main()

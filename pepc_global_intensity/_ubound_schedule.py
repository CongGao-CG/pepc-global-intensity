from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
PI_UBOUND_SUMMARY_PATH = DATA_DIR / "pi_historical_ubound_and_ssp585_quantiles_by_basin_model.tsv"

SSP585_UBOUND_WINDOWS = (
    (2015, 2044, "ssp585_2015_2049_minus_ubound_kt"),
    (2045, 2069, "ssp585_2040_2074_minus_ubound_kt"),
    (2070, 2099, "ssp585_2065_2099_minus_ubound_kt"),
)


@lru_cache(maxsize=1)
def _load_pi_ubound_summary() -> pd.DataFrame:
    if not PI_UBOUND_SUMMARY_PATH.is_file():
        raise FileNotFoundError(f"Missing PI ubound summary: {PI_UBOUND_SUMMARY_PATH}")
    return pd.read_csv(PI_UBOUND_SUMMARY_PATH, sep="\t", keep_default_na=False)


def get_intensity_ubound_schedule(
    basin: str,
    model: str,
    experiment: str,
    variant: str,
    intensity_ubound: float,
) -> tuple[tuple[int, int, float], ...]:
    if experiment == "historical":
        return ((1980, 2014, float(intensity_ubound)),)

    summary = _load_pi_ubound_summary()
    rows = summary[
        (summary["basin"] == basin)
        & (summary["model"] == model)
        & (summary["ssp585_variant"] == variant)
    ]
    if rows.empty:
        raise RuntimeError(
            f"Missing ssp585 ubound adjustment for basin={basin}, model={model}, variant={variant}."
        )
    if len(rows) > 1:
        raise RuntimeError(
            f"Multiple ssp585 ubound adjustments found for basin={basin}, model={model}, variant={variant}."
        )

    row = rows.iloc[0]
    row_ubound = float(row["ubound_kt"])
    if not np.isclose(row_ubound, intensity_ubound):
        raise RuntimeError(
            f"Ubound mismatch for basin={basin}, model={model}, variant={variant}: "
            f"dict={intensity_ubound}, summary={row_ubound}."
        )

    schedule = []
    for start_year, end_year, column in SSP585_UBOUND_WINDOWS:
        diff = row[column]
        if pd.isna(diff):
            raise RuntimeError(
                f"Missing ssp585 ubound difference in column '{column}' for "
                f"basin={basin}, model={model}, variant={variant}."
            )
        schedule.append((start_year, end_year, float(intensity_ubound + float(diff))))
    return tuple(schedule)

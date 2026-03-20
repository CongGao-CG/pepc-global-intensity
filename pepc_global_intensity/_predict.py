from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from ._softmax import softmax
from ._ubounds import get_intensity_ubounds_dict
from ._reach import get_intensity_reach_dict
from ._ubound_schedule import get_intensity_ubound_schedule


DATA_DIR = Path(__file__).resolve().parent / "data"

N_STATES = 3
WIND_TO_MS = 0.514
MIN_PRE_WIND = 10.0
exponent_upbound = 2


@lru_cache(maxsize=None)
def _load_parameters_for_basin(basin: str) -> Dict[str, Any]:
    hmm_path = DATA_DIR / f"BEST_HMM_{N_STATES}_{basin}.pkl"
    with open(hmm_path, "rb") as f:
        raw_hmm = pickle.load(f)
    hmm_params = {
        "States": list(raw_hmm["States"]),
        "startCoefs": np.asarray(raw_hmm["startCoefs"], dtype=float),
        "transCoefs": {st: np.asarray(mat, dtype=float) for st, mat in raw_hmm["transCoefs"].items()},
        "emissionParams": {
            st: {
                "coefs": np.asarray(params["coefs"], dtype=float),
                "sd": float(params["sd"]),
            }
            for st, params in raw_hmm["emissionParams"].items()
        },
    }

    coeffs_path = DATA_DIR / f"intensity_land_exp-predictors-dry-rough_{basin}_coeffs.npy"
    sigma_path = DATA_DIR / f"intensity_land_exp-predictors-dry-rough_{basin}_noise-sigma.npy"
    land_coeffs = np.load(coeffs_path)
    land_sigma = np.load(sigma_path)

    return {
        "hmm": hmm_params,
        "land": {"coeffs": land_coeffs, "sigma": land_sigma},
    }


def _sample_state(
    start_coefs: np.ndarray,
    trans_coefs: Dict[str, np.ndarray],
    states: Tuple[str, ...],
    features: np.ndarray,
    prev_state: str | None,
    rng: np.random.Generator,
) -> str | None:
    if np.any(np.isnan(features)):
        return None
    if prev_state is None:
        logits = start_coefs @ features
    else:
        prev_matrix = trans_coefs.get(prev_state)
        if prev_matrix is None:
            return None
        logits = prev_matrix @ features
    probs = softmax(logits)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return None
    probs = probs / total
    return str(rng.choice(states, p=probs))


def _process_group(
    group: pd.DataFrame,
    basin_params: Dict[str, Any],
    seed: int,
    intensity_ubound: float,
    intensity_ubound_schedule: tuple[tuple[int, int, float], ...],
    intensity_reach: float,
):
    ordered_idx = group.index.to_numpy()
    if ordered_idx.size == 0:
        return (
            ordered_idx,
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([], dtype=object),
        )

    step = group["step"].to_numpy()
    years = group["year"].to_numpy(dtype=int, copy=False)
    mode = group["mode"].to_numpy()
    ild = group["ILD"].to_numpy()
    grd = group["grd"].to_numpy()
    ut = group["ut"].to_numpy()
    pi_vals = group["pi"].to_numpy()
    shr = group["shr"].to_numpy()
    rh600 = group["rh600"].to_numpy()
    ocean = group["ocean"].to_numpy()
    ocean_next = group["ocean_next"].to_numpy()
    mean_stl1 = group["mean_stl1"].to_numpy()
    mean_fsr = group["mean_fsr"].to_numpy()
    mean_swvl1 = group["mean_swvl1"].to_numpy()

    pre_wind = group["pre_wind"].to_numpy(copy=True)
    pre_delta_wind_b = group["pre_delta_wind_b"].to_numpy(copy=True)
    pre_delta_wind_f = group["pre_delta_wind_f"].to_numpy(copy=True)
    pre_ocn = group["pre_ocn"].to_numpy(copy=True)
    states = np.full(ordered_idx.shape, None, dtype=object)
    effective_ubound = np.full(ordered_idx.shape, float(intensity_ubound), dtype=float)
    for start_year, end_year, ubound_value in intensity_ubound_schedule:
        mask = (years >= start_year) & (years <= end_year)
        effective_ubound[mask] = ubound_value

    with np.errstate(divide="ignore", invalid="ignore"):
        ocn_base = -0.01 * np.power(grd * 100.0, -0.4) * ild * ut * pi_vals

    hmm_params = basin_params["hmm"]
    hmm_states = tuple(hmm_params["States"])
    start_coefs = np.asarray(hmm_params["startCoefs"], dtype=float)
    trans_coefs = {st: np.asarray(mat, dtype=float) for st, mat in hmm_params["transCoefs"].items()}
    emission_params = {
        st: {
            "coefs": np.asarray(params["coefs"], dtype=float),
            "sd": float(params["sd"]),
        }
        for st, params in hmm_params["emissionParams"].items()
    }
    land_params = basin_params.get("land", {})
    land_coeffs = land_params.get("coeffs")

    local_rng = np.random.default_rng(int(seed))
    prev_pos = None
    prev_state = None
    halt = False
    for pos in range(ordered_idx.size):
        if halt:
            pre_delta_wind_f[pos] = np.nan
            pre_delta_wind_b[pos] = np.nan
            pre_wind[pos] = np.nan
            pre_ocn[pos] = np.nan
            states[pos] = None
            continue

        if step[pos] > 0 and prev_pos is not None:
            prev_delta = pre_delta_wind_f[prev_pos]
            pre_delta_wind_b[pos] = prev_delta
            prev_wind = pre_wind[prev_pos]
            if np.isnan(prev_delta) or np.isnan(prev_wind):
                pre_wind[pos] = np.nan
                halt = True
                prev_pos = pos
                states[pos] = None
                continue
            new_pre_wind = prev_wind + prev_delta
            if new_pre_wind < MIN_PRE_WIND + 1:
                pre_wind[pos] = MIN_PRE_WIND + 1
                halt = True
                prev_pos = pos
                states[pos] = None
                continue
            pre_wind[pos] = new_pre_wind

        mode_val = mode[pos]
        if mode_val == "ocean":
            base = ocn_base[pos]
            wind_val = pre_wind[pos]
            if np.isnan(base) or np.isnan(wind_val):
                pre_ocn[pos] = np.nan
            else:
                denom = wind_val * WIND_TO_MS
                if denom == 0:
                    pre_ocn[pos] = np.nan
                else:
                    exponent = base / denom
                    ocn = 1.0 - 0.87 * np.exp(exponent)
                    pre_ocn[pos] = float(np.clip(ocn, 0.0, 1.0)) if not np.isnan(ocn) else np.nan

        pred = np.nan
        current_state = prev_state

        if mode_val == "ocean":
            features = np.asarray(
                (
                    1.0,
                    pre_delta_wind_b[pos],
                    pre_wind[pos],
                    pi_vals[pos],
                    shr[pos],
                    rh600[pos],
                    pre_ocn[pos],
                ),
                dtype=float,
            )
            current_state = _sample_state(
                start_coefs,
                trans_coefs,
                hmm_states,
                features,
                prev_state,
                local_rng,
            )
            if current_state is not None:
                emission = emission_params[current_state]
                pred = float(np.dot(emission["coefs"], features))
                sd = emission["sd"]
                if not np.isnan(pred) and sd > 0.0:
                    pred = float(pred + local_rng.normal(loc=0.0, scale=sd))
        elif mode_val == "land":
            if land_coeffs is not None:
                features = np.asarray(
                    [
                        (
                            pre_wind[pos],
                            mean_stl1[pos],
                            mean_fsr[pos],
                            mean_swvl1[pos],
                        )
                    ],
                    dtype=float,
                )
                logratio_pred = float(land_coeffs[0] + np.dot(land_coeffs[1:], features[:, 1:].ravel()))
                pred = np.exp(logratio_pred) * (features[:, 0] - MIN_PRE_WIND) + MIN_PRE_WIND - features[:, 0]
                pred = float(np.asarray(pred).ravel()[0])
        elif mode_val == "not":
            pred = np.nan

        states[pos] = current_state
        current_intensity_ubound = effective_ubound[pos]
        if pred > 0 and pre_wind[pos] > current_intensity_ubound - intensity_reach:
            damping = max(0, 1 - (pre_wind[pos] / current_intensity_ubound) ** exponent_upbound)
            pred = pred * damping
        pre_delta_wind_f[pos] = pred
        prev_pos = pos
        prev_state = current_state

    return ordered_idx, pre_wind, pre_delta_wind_b, pre_delta_wind_f, pre_ocn, states


def predict_intensity(
    basin: str,
    df: pd.DataFrame,
    experiment: str,
    model: str,
    variant: str,
    seed: int | None = None,
) -> pd.DataFrame:
    """Predict tropical cyclone intensity for a single storm.

    Parameters
    ----------
    basin : str
        Basin identifier (e.g. "NA", "WNP", "ENP", "AS", "BoB", "SI", "SP").
    df : pd.DataFrame
        Pre-prepared DataFrame for one storm. Required columns:
        step, time, year, mode, ILD, grd, ut, pi, shr, rh600,
        ocean, ocean_next, mean_stl1, mean_fsr, mean_swvl1.
    experiment : str
        CMIP6 experiment ("historical", "ssp585", or "ssp245").
    model : str
        CMIP6 model name (used for SSP ubound schedule lookup).
    variant : str
        CMIP6 variant label (used for SSP ubound schedule lookup).
    seed : int or None
        Random seed for reproducibility. If None, a random seed is used.

    Returns
    -------
    pd.DataFrame
        Copy of the input with added columns: pre_wind, pre_delta_wind_b,
        pre_delta_wind_f, pre_ocn, state.
    """
    basin_params = _load_parameters_for_basin(basin)
    intensity_ubound = get_intensity_ubounds_dict(basin)
    intensity_ubound_schedule = get_intensity_ubound_schedule(
        basin, model, experiment, variant, intensity_ubound,
    )
    intensity_reach = get_intensity_reach_dict(basin)

    result = df.copy()
    cols = ["pre_wind", "pre_delta_wind_b", "pre_ocn", "pre_delta_wind_f"]
    result[cols] = np.nan
    result.loc[result["step"] == 0, "pre_delta_wind_b"] = 5.0
    result.loc[result["step"] == 0, "pre_wind"] = 35.0
    result["state"] = pd.Series([None] * len(result), dtype=object, index=result.index)

    result = result.sort_values("time").reset_index(drop=True)

    if seed is None:
        seed = int(np.random.default_rng().integers(0, np.iinfo(np.int64).max))

    ordered_idx, pre_wind, pre_delta_wind_b, pre_delta_wind_f, pre_ocn, states = _process_group(
        result,
        basin_params,
        seed,
        intensity_ubound,
        intensity_ubound_schedule,
        intensity_reach,
    )

    result.loc[ordered_idx, "pre_wind"] = pre_wind
    result.loc[ordered_idx, "pre_delta_wind_b"] = pre_delta_wind_b
    result.loc[ordered_idx, "pre_delta_wind_f"] = pre_delta_wind_f
    result.loc[ordered_idx, "pre_ocn"] = pre_ocn
    result.loc[ordered_idx, "state"] = states

    return result

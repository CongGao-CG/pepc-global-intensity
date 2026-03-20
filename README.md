# PepC-Global Intensity

A Python package for predicting tropical cyclone intensity using pre-trained Input-Output Hidden Markov Models.

## Installation

```bash
pip install --upgrade pepc-global-intensity
```

Or from source:

```bash
git clone https://github.com/CongGao-CG/pepc-global-intensity.git
cd pepc-global-intensity
pip install .
```

## Quick Start

```python
import pandas as pd
from pepc_global_intensity import predict_intensity

# Create DataFrame for a single storm with required columns
df = pd.DataFrame({
    "step":       [0, 1, 2],
    "time":       pd.to_datetime(["2000-07-01", "2000-07-01 06:00", "2000-07-01 12:00"]),
    "year":       [2000, 2000, 2000],
    "mode":       ["ocean", "ocean", "ocean"],
    "ILD":        [50.0, 50.0, 50.0],
    "grd":        [1e-4, 1e-4, 1e-4],
    "ut":         [5.0, 5.0, 5.0],
    "pi":         [70.0, 70.0, 70.0],
    "shr":        [10.0, 10.0, 10.0],
    "rh600":      [0.7, 0.7, 0.7],
    "ocean":      [True, True, True],
    "ocean_next": [True, True, True],
    "mean_stl1":  [0.0, 0.0, 0.0],
    "mean_fsr":   [0.0, 0.0, 0.0],
    "mean_swvl1": [0.0, 0.0, 0.0],
})

# Predict intensity for a specific basin and experiment
result = predict_intensity(
    basin="NA",  # North Atlantic
    df=df,
    experiment="historical",
    model="ACCESS-ESM1-5",
    variant="r1i1p1f1",
    seed=42,
)

print(result[["step", "time", "pre_wind", "state"]])
```

## Available Basins

| Basin Code | Name                  | Latitude Range    | Longitude Range     |
|------------|-----------------------|-------------------|---------------------|
| `AS`       | Arabian Sea           | 5° to 22.5°N      | 50° to 77.5°E       |
| `BoB`      | Bay of Bengal         | 5° to 22.5°N      | 80° to 100°E        |
| `WNP`      | Western North Pacific | 5° to 30°N        | 102.5°E to 180°     |
| `ENP`      | Eastern North Pacific | 5° to 25°N        | 177.5° to 75°W      |
| `NA`       | North Atlantic        | 5° to 30°N        | 97.5° to 2.5°W      |
| `SI`       | South Indian          | 30° to 5°S        | 20° to 145°E        |
| `SP`       | South Pacific         | 30° to 5°S        | 147.5°E to 100°W    |

## API Reference

### `predict_intensity(basin, df, experiment, model, variant, seed)`

Predict tropical cyclone intensity using Input-Output Hidden Markov Models.

**Parameters:**
- `basin` (str): Basin name (one of: 'AS', 'BoB', 'WNP', 'ENP', 'NA', 'SI', 'SP')
- `df` (pd.DataFrame): Pre-prepared data for a single storm with required columns (see Input Variables table below)
- `experiment` (str): Experiment type — 'historical' or 'ssp585'
- `model` (str): CMIP6 model name (used for ssp585 upper-bound schedule lookup)
- `variant` (str): CMIP6 variant label (used for ssp585 upper-bound schedule lookup)
- `seed` (int or None): Random seed for reproducibility

**Returns:**
- `pd.DataFrame`: Copy of the input DataFrame with added columns:
  - `pre_wind`: Predicted wind speed
  - `pre_delta_wind_b`: Predicted backward wind change
  - `pre_delta_wind_f`: Predicted forward wind change
  - `pre_ocn`: Predicted ocean state
  - `state`: Hidden state classification

**Raises:**
- `ValueError`: If basin is invalid, experiment is not 'historical' or 'ssp585', or required columns are missing from df

### `get_basin_names()`

Get the list of valid basin names.

**Returns:**
- `list[str]`: List of 7 basin names

### `BASINS`

List of valid basin names.

## Input Variables

| Variable     | Description                          | Typical Units |
|--------------|--------------------------------------|---------------|
| step         | Track point index within the storm (0 at genesis) | —             |
| time         | Timestamp                            | datetime      |
| year         | Calendar year                        | —             |
| mode         | Surface type: 'ocean', 'land', or 'not' | —          |
| ILD          | Isothermal layer depth               | m             |
| grd          | Ocean temperature gradient           | K m^−1        |
| ut           | Storm translation speed              | m s^−1        |
| pi           | Potential intensity                  | kt            |
| shr          | Vertical wind shear between 200 hPa and 850 hPa | m s^−1        |
| rh600        | Relative humidity at 600 hPa         | %             |
| ocean        | ocean fraction at the current position | —           |
| ocean_next   | ocean fraction at the next 6-hour position | —       |
| mean_stl1    | Land-weighted soil temperature       | K             |
| mean_fsr     | Land-weighted surface roughness length | m           |
| mean_swvl1   | Land-weighted soil water volume      | m^3 m^−3      |

## Model Details

This package contains pre-trained Input-Output Hidden Markov Models (IOHMM) for 7 tropical cyclone basins. The models were pre-trained on ERA5 monthly data using multiple environmental predictors known to influence tropical cyclone intensity:

1. **Ocean coupling (ILD, grd)**: Ocean thermal structure affecting ocean feedback
2. **Storm motion (ut)**: Translation speed affecting ocean feedback
3. **Potential intensity (pi)**: Theoretical upper bound on storm strength
4. **Wind shear (shr)**: Vertical wind shear affecting intensification
5. **Mid-level humidity (rh600)**: Environmental moisture affecting intensification
6. **Land effects (mean_fsr, mean_stv1, mean_swvl1)**: Land and soil conditions affecting over-land decay

Each basin has its own pre-trained IOHMM parameters and land-mode regression coefficients.

## License

MIT License

## Citation

If you use this package in your research, please cite:

Gao, Cong, Ning Lin. "PepC-Global: A Basin-Tuned Probabilistic Tropical Cyclone Model with Enhanced Out-of-Sample Skill and Climate-Sensitive Over-Land Decay". Journal of Advances in Modeling Earth Systems (JAMES), under review.

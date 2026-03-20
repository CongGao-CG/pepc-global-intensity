# pepc-global-intensity

Predict tropical cyclone intensity using pre-trained Input-Output Hidden Markov Models.

## Installation

```bash
pip install pepc-global-intensity
```

## Usage

```python
import pandas as pd
from pepc_global_intensity import predict_intensity

# DataFrame for a single storm with required columns:
#   step, time, year, mode, ILD, grd, ut, pi, shr, rh600,
#   ocean, ocean_next, mean_stl1, mean_fsr, mean_swvl1
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

result = predict_intensity(
    basin="NA",
    df=df,
    experiment="historical",
    model="ACCESS-ESM1-5",
    variant="r1i1p1f1",
    seed=42,
)
```

## Parameters

- **basin**: `str` — one of `"AS"`, `"BoB"`, `"WNP"`, `"ENP"`, `"NA"`, `"SI"`, `"SP"`
- **df**: `pandas.DataFrame` — pre-prepared data for a single storm with columns:
  - `step` — track point index within the storm (0 at genesis)
  - `time` — timestamp
  - `year` — calendar year
  - `mode` — `"ocean"`, `"land"`, or `"not"`
  - `ILD` — isothermal layer depth (m)
  - `grd` — ocean temperature gradient
  - `ut` — storm translation speed (m/s)
  - `pi` — potential intensity (kt)
  - `shr` — wind shear (m/s)
  - `rh600` — relative humidity at 600 hPa
  - `ocean` — whether the current point is over ocean
  - `ocean_next` — whether the next point is over ocean
  - `mean_stl1` — land-weighted soil temperature
  - `mean_fsr` — land-weighted fraction of sunshine radiation
  - `mean_swvl1` — land-weighted soil water volume
- **experiment**: `str` — `"historical"` or `"ssp585"`
- **model**: `str` — CMIP6 model name (used for ssp585 upper-bound schedule lookup)
- **variant**: `str` — CMIP6 variant label (used for ssp585 upper-bound schedule lookup)
- **seed**: `int` or `None` — random seed for reproducibility

## Returns

- `pandas.DataFrame` — copy of the input with added columns: `pre_wind`, `pre_delta_wind_b`, `pre_delta_wind_f`, `pre_ocn`, `state`

## Model Weights

Model weights (HMM parameters and land-mode regression coefficients) are bundled with the package.

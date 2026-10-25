# Estimating gas-turbine compressor degradation from sensor measurements

MMA3001 Numerical Methods and Machine Learning: individual project.

## Problem

Gas-turbine compressors lose performance over time as their blades foul and
erode. This project estimates the **compressor degradation (decay) coefficient**
of a naval gas turbine from routine sensor measurements (temperatures,
pressures, rotational speeds, torques and fuel flow).

Linear regression, decision-tree regression and support-vector regression (SVR)
are compared against a mean-prediction baseline using training, validation and
test sets, and evaluated with MAE, RMSE and R². The project also investigates
how sensor selection and model settings affect accuracy and runtime.

The engineering aim is to assess how reliably these measurements indicate
compressor condition within the simulated conditions.

## Data

**Source:** Coraddu, A., Oneto, L., Ghio, A., Savio, S., Anguita, D. and
Figari, M. (2014). *Condition Based Maintenance of Naval Propulsion Plants*
[Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5K31K>. Licensed under CC BY 4.0.

The data is included in `data/raw/` with attribution. It comes from a
validated numerical simulator of a frigate's gas-turbine propulsion plant:
11,934 steady-state cases covering a full grid of 9 ship speeds × 51
compressor decay levels (0.950 to 1.000) × 26 turbine decay levels (0.975 to 1.000).

| | Details |
|---|---|
| **Inputs** | 16 sensor measurements; 4 are removed as redundant (below), leaving 12 |
| **Output** | Compressor decay coefficient `kMc`, dimensionless, 0.95 (most degraded) to 1.00 (new) |
| **Not used** | Turbine decay coefficient `kMt`: it would not be known in practice |

**Data issues found and handled:**

- Compressor inlet temperature (T1) and pressure (P1) are constant, so they are removed.
- Port and starboard propeller torque are identical, and lever position and ship
  speed map one-to-one, so one of each pair is removed.
- Temperatures are documented in °C but are in kelvin (the inlet is a constant
  288 K = 15 °C).
- The dataset's own `README.txt` is titled "Human Activity Recognition Using
  Smartphones", a copy-paste error.

## Results (locked test set, 2,387 cases)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Mean baseline | 0.01272 | 0.01467 | 0.000 |
| Linear regression | 0.00529 | 0.00651 | 0.803 |
| Decision tree (unlimited depth) | 0.00097 | 0.00151 | 0.989 |
| **SVR** (C = 100, ε = 0.0001, γ = scale) | **0.00019** | **0.00023** | **0.9998** |

Further results (tuning, sensor selection, extrapolation) are in `results/`
and `figures/`, produced by `notebooks/02_results.ipynb`.

## Repository structure

```
├── data/raw/            UCI dataset (data.txt, Features.txt, README.txt)
├── docs/                HTML code documentation and pytest test report
├── figures/             Report figures (PNG)
├── notebooks/
│   ├── 01_explore_data.ipynb   Data loading, split, default models, tuning
│   ├── 02_results.ipynb        Sensor selection, test results, extrapolation, figures
│   └── 03_performance.ipynb    Profiling, memory, scalability, stability, noise, repeatability
├── results/             CSV results tables
├── src/degradation/     Source code (see docs/ for full documentation)
├── tests/               pytest tests
├── pyproject.toml       pytest configuration
├── requirements.txt     Exact package versions
└── LICENSE              MIT licence (code); the data is CC BY 4.0
```

## Setup (Windows)

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Running

| Task | Command / action |
|---|---|
| Run the tests | `python -m pytest -v` |
| Regenerate the test report | `python -m pytest --html=docs/test_report.html --self-contained-html` |
| Regenerate the HTML documentation | `pdoc --docformat numpy src/degradation -o docs` |
| Reproduce the results | Open the notebooks in VS Code and click **Run All** (notebooks 01, 02, then 03) |

All random choices use a fixed seed (`RANDOM_STATE = 42`), so results are reproducible.

## Use of AI

Claude (Anthropic) was used during this project for planning, code writing,
explanations and drafting. All use is disclosed and reflected on in the AI
section of the report.

## Licence

Code: MIT (see `LICENSE`). Data: CC BY 4.0, © the dataset authors (see Data).

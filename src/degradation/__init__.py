"""Estimating gas-turbine compressor degradation from sensor measurements.

MMA3001 individual project.  The package compares a mean-prediction
baseline, linear regression, decision-tree regression and support-vector
regression (SVR) for estimating the compressor decay coefficient of a naval
gas turbine from 16 simulated sensor measurements (UCI dataset 316).

Modules, in workflow order
--------------------------
- `degradation.data`: load and validate the data; remove redundant sensors.
- `degradation.splitting`: locked test set and k-fold cross-validation.
- `degradation.models`: the four models, the MAE/RMSE/R-squared metrics and
  cross-validated scoring.
- `degradation.tuning`: grid search of model settings.
- `degradation.selection`: sensor selection and conditioning.
- `degradation.evaluation`: final test-set scores and the extrapolation check.
- `degradation.figures`: report figures.
"""

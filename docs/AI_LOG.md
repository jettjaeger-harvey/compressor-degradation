# AI Use Log: MMA3001 Individual Project

*Estimating gas-turbine compressor degradation from sensor measurements*

**Tool:** Claude (Anthropic), via claude.ai. No other generative AI was used.

**Key:** *AI* = what Claude did · *Me* = what I did, decided or checked · *AI error* = a mistake or unhelpful output I noticed or that was corrected.

## 1. Choosing the project

- **AI:** Summarised the project brief and compared the options (apple-tree reconstruction, frozen-meat images, a heat-conduction simulation, and Dataset 2: building occupancy).
- **Me:** Chose Dataset 2 (building occupancy).

## 2. Setting up the environment

- **AI:** Gave step-by-step instructions for installing VS Code, Python, Git and the VS Code extensions, creating the GitHub repository, the folder structure and the virtual environment, and making the first commit.
- **Me:** Did the installation and setup myself. Fixed errors along the way with Claude's help: Git not recognised, PowerShell blocking scripts, and the virtual environment breaking after I moved the project out of OneDrive and later renamed the folder.
- **AI error:** Some instructions skipped steps for a beginner (for example, a keyboard shortcut that zoomed instead of opening the terminal on my layout).

## 3. First dataset: building occupancy

- **AI:** Audited the occupancy and environmental files. It found about 76% of the environmental rows were frozen repeats of old readings, the environmental sensors were not in the same rooms as the occupancy sensors, and there were false night-time detections. It produced a cleaned hourly Excel file and cleaning code.
- **Me:** Looked through the data and the EdStem comments, and decided the data was too poor for my question.
- **AI error:** It first said to remove all long "occupied" periods; on checking, most were real busy days, so the rule was narrowed.

## 4. Changing project

- **Me:** Found the gas-turbine compressor degradation project on the unit forum (EdStem) and wrote my own problem statement. Claude suggested additions; I chose to keep my statement unchanged. Renamed the GitHub repository and project folder and added the new dataset.

## 5. Data loading (`data.py`)

- **AI:** Wrote `data.py`, which names every sensor with its unit; stops with a clear error if the file is missing, has the wrong number of columns, contains non-numbers or has out-of-range targets; removes the 4 sensors that are useless by definition; excludes turbine decay as an input; and documents every function with a docstring for pdoc. It also identified that the temperatures are in kelvin, not °C as documented.
- **Me:** Ran a "smoke test" in a notebook to check `data.py` on my dataset (11,934 rows, 4 sensors removed, 12 inputs kept).
- **AI error:** It reported a condition number of about 10¹⁷ for the inputs, which was before the duplicate sensors were removed; with the 12 sensors actually used it is 842.

## 6. Tests for data loading (`test_data.py`)

- **AI:** Wrote six tests: constant column flagged, copied column flagged, independent columns kept, missing file raises an error, wrong number of columns raises an error, out-of-range target raises an error.
- **Me:** Checked the tests by deliberately breaking the code and confirming the matching test failed, which showed both `data.py` and the tests were working.
- **AI error:** One of its tests was passing for the wrong reason (pandas raised an error before the code's own check), found when the code was broken deliberately; the test was tightened.
- **AI error:** It suggested a log entry stating I had done a check that it had actually done. I noticed and questioned it; from then on, log entries were only written after I did the check myself.

## 7. Splitting the data (`splitting.py`)

- **Me:** Decided to use 5-fold cross-validation, as discussed in class, with 20% of the data held out unseen for final testing.
- **AI:** Wrote `splitting.py` and `test_splitting.py` to do this.
- **Me:** Ran the tests; checked in a notebook that no row was in both the development and test sets; then replaced the split with `return X, X.iloc[:20], y, y.iloc[:20]` (putting all rows in the development set) and confirmed the leak test failed, before restoring the code.
- **AI error:** It warned that random cross-validation would be over-optimistic for this gridded data. When I asked whether cross-validation works in this case, it tested the claim: the warning was wrong for interpolation, but right for extrapolation.

## 8. Models and metrics (`models.py`)

- **Me:** Chose to keep the three models from my problem statement rather than adding a neural network.
- **AI:** Wrote the model and metric code.
- **Me:** Checked the metric formulas against a hand calculation (MAE 0.5, RMSE 0.707, R² 0.6) and confirmed the test fails when RMSE is deliberately computed wrongly. Found that default SVR performed no better than the baseline, because its ε (0.1) exceeds the whole target range (0.05). I later asked whether default SVR predicts a constant or a linear function; inspecting the fitted model confirmed a constant (0 support vectors, every prediction 0.975).

## 9. Tuning (`tuning.py`)

- **AI:** Wrote the tuning code and ran an extended check beyond the edge of the grid.
- **Me:** Ran the grid search in my notebook and confirmed tuning reduced SVR's cross-validated MAE from 0.0128 (default) to 0.00015, and that the best decision tree had unlimited depth because the simulated data is noise-free. Questioned why the grid had so few values; the answer (cost multiplies with each value, and the top settings were within the fold-to-fold spread) was supported with evidence.
- **AI error:** Its instructions did not clearly tell me to create the `results` folder, which caused an error when saving.

## 10. Sensor selection, final results and figures

- **AI:** Wrote the sensor-selection, evaluation and figure code and ran the slow SVR subset study.
- **Me:** Reran the results notebook and confirmed the test-set scores matched (SVR MAE 0.00019, R² 0.9998).
- **AI error:** Its first SVR sensor-selection run would have taken hours, because SVR becomes slower with fewer sensors; it changed approach and ranked sensors with the decision tree instead.

## 11. Repository documentation

- **AI:** Drafted the README and set up pdoc and pytest-html.
- **Me:** Generated the HTML documentation and the test report myself, made the dataset part of the repository (CC BY 4.0), and checked the repository against the brief's list of required items.

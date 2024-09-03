import os
import pathlib

data_path = None # Set Data Path Here
raw_data_path = data_path / "RAW"
parsed_data_path = data_path / "Parsed"
csv_path = parsed_data_path / "CSV"


### Training Paths ###
project_path = None # Set Project Path Here
checkpoints_path = project_path / "Checkpoints"
logs_path = project_path / "Logs"
figures_path = project_path / "Figures"
models_path = project_path / "Models"
models_2_path = data_path / "Models"
results_path = project_path / "Results"
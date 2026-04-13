#!/bin/bash

echo "Building C3+ Project Architecture..."

# 1. Create all the directories (the -p flag ensures parent directories are made automatically)
mkdir -p config
mkdir -p models/objects
mkdir -p sim
mkdir -p control
mkdir -p utils

# 2. Create the root level files
touch main.py
touch environment.yml

# 3. Create the configuration files
touch config/sim_settings.yaml
touch config/admm_params.yaml

# 4. Create the model files
touch models/objects/push_box.sdf

# 5. Create the simulation files
touch sim/__init__.py
touch sim/plant_builder.py
touch sim/systems.py

# 6. Create the control algorithm files
touch control/__init__.py
touch control/controller.py
touch control/admm_solver.py
touch control/lcs_formulator.py

# 7. Create the utility files
touch utils/__init__.py
touch utils/projections.py
touch utils/qp_solvers.py
touch utils/meshcat_utils.py

echo "Directory structure successfully created!"
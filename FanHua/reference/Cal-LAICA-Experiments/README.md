# Cal-LAICA Experimental Code

This repository contains the experimental code for the thesis algorithm **Cal-LAICA**. In the recommendation-system experiments, the same calibrated Q-learning component is named **CAL-QL**.

The project is organized as a single codebase with shared, portable paths.

### Project structure

`run_cal_laica.py` runs the Cal-LAICA GridWorld changing-action experiment.

`run_recsys_calql.py` runs the recommendation-system CAL-QL experiments on ML-1M or AB-1M.

`cal_laica/` contains the GridWorld environment, Cal-LAICA agent, runner, and utilities.

`recsys_calql/` contains the recommendation environment, CAL-QL agent, data loaders, runner, and utilities.

`data/ml-1m/` contains the MovieLens-1M data files.

`data/ab-1m/` contains the AB-1M interaction, book, and knowledge-graph files.

Generated logs, curves, checkpoints, and NumPy result files are written to `outputs/`.

### How to run

Run commands from this folder:

```bash
cd /path/to/Cal-LAICA-Experiments
```

#### Cal-LAICA GridWorld

```bash
python run_cal_laica.py
```

#### CAL-QL on ML-1M

```bash
python run_recsys_calql.py --experiment ML-1M
```

#### CAL-QL on AB-1M

```bash
python run_recsys_calql.py --experiment AB-1M
```

The recommendation entry point chooses the dataset path automatically:

```text
ML-1M -> data/ml-1m
AB-1M -> data/ab-1m
```

You can override the dataset location with `--data_path`. Relative paths are resolved from the project root, so the code can be moved to another computer without hard-coded local paths.

## Environment setup

### 1. Create and activate a Python environment.

```bash
conda create -n cal-laica python=3.11
conda activate cal-laica
```

### 2. Install dependencies.

```bash
pip install -r requirements.txt
pip install numpy pandas matplotlib torch tqdm pyyaml scikit-learn
```

### 3. Check the data layout.

Before running the recommendation experiments, ensure the following files exist:

```text
data/ml-1m/ratings.dat
data/ml-1m/users.dat
data/ml-1m/movies.dat

data/ab-1m/ab_1_20.dat
data/ab-1m/books.dat
data/ab-1m/triplet.dat
data/ab-1m/ab2fb.txt
```

#!/bin/bash

#SBATCH --job-name="pt_index"
#SBATCH --time=03:20:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load python
module load py-pip
module load openjdk
# module load cuda

# Set up virtual environment
# python -m venv env
# source env/bin/activate

# Install dependencies
python -m pip install --user python-terrier==0.10.0 fast-forward-indexes==0.2.0 jupyter ipywidgets

# Run the experiment
srun python index.py > prints.txt

# Exit the virtual environment
# deactivate

# Remove the virtual environment
# rm -r env/

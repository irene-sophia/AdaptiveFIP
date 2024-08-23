#!/bin/bash -l

#SBATCH --job-name="MH_reopt"
#SBATCH --time=10:00:00
#SBATCH --partition=compute

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0

#SBATCH --account=research-tpm-mas

module load 2022r2
module load openmpi
module load python
module load py-mpi4py


srun python runfile_grid.py $1 $2 $3
#!/bin/bash
#SBATCH -o ./slurm_output/R-%x.%j.out
#SBATCH -t 8:00:00          # walltime = 1 hours and 30 minutes
#SBATCH -n 1              # one CPU (hyperthreaded) cores
#SBATCH -p normal
#SBATCH --mem=32G
#SBATCH --array=1-6%30
SBATCH --mail-type=NONE


# Execute commands to run your program here, taking Python for example,
cd /om2/user/c_tang/jazlab/rnn/src/submission
python fig4b.py $SLURM_ARRAY_TASK_ID


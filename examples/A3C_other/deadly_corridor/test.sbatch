#!/bin/bash
#
#SBATCH --job-name=a3c-exp-36 # Job name for tracking
#SBATCH --partition=cpu-batch  # Partition you wish to use (see above for list)
#SBATCH --cpus-per-task=20     # Number of CPU threads used by your job, set this upto 40 as required.
#SBATCH --mem=60000            # 60GB RAM
#SBATCH --exclusive=mcs        # Exclusive mode, only this job will run
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##
# python3 -c "print('hello')"
python deadly_corridor.py
## Other examples:
## A Python program
# python3 my_program.py
#
## Excutable program (e.g. with shebang or written in C/C++)
# ~/myprogram
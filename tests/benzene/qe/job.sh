#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="aiida-3036"
#SBATCH --get-user-env
#SBATCH --output=_scheduler-stdout.txt
#SBATCH --error=_scheduler-stderr.txt
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

### computer prepend_text start ###
#SBATCH --partition=normal
#SBATCH --account=s1267
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
ulimit -s unlimited
### computer prepend_text end ###

module load daint-gpu
module load QuantumESPRESSO


'srun' '-n' '64' '/apps/daint/UES/jenkins/7.0.UP03/21.09/daint-gpu/software/QuantumESPRESSO/7.2-CrayNvidia-21.09/bin/projwfc.x' < 'aiida.in' > 'aiida.out'

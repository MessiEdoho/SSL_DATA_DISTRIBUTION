#!/bin/bash -l
#SBATCH --job-name=audit_manifests
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=csgpu
## Audit is CPU-only (manifest reads). If the partition requires a GPU
## allocation, uncomment the next line:
##SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie
#SBATCH --output=/home/people/22206468/slurm-audit_manifests-%j.out

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "----- CPU allocation -----"
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-unset}"
echo "nproc (visible)     : $(nproc)"
echo "--------------------------"

# Activate environment (reuse the baseline conda env).
module purge
module load anaconda3
conda activate torch_v100_py310

# Adjust to the actual repo location on the cluster if different.
cd ~/SSL_DATA_DISTRIBUTION

# Add --scan-folders to also count .npy files physically on disk (slow).
python scripts/audit_manifests.py

echo "===== JOB END ====="
date

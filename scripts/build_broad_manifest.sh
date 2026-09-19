#!/bin/bash -l
#SBATCH --job-name=build_broad
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=csgpu
## CPU-only; memory-heavy (loads the full ~28.7M-record manifest). If the
## partition requires a GPU allocation, uncomment:
##SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie
#SBATCH --output=/home/people/22206468/slurm-build_broad-%j.out

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/SSL_DATA_DISTRIBUTION

# [TBD-4] set the broadening level R (non-ictal per ictal); default 6.
python scripts/build_broad_manifest.py --target-nonictal-per-ictal 6

echo "===== JOB END ====="
date

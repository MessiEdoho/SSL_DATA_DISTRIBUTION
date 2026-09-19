#!/bin/bash -l
#SBATCH --job-name=ssl_pretrain_Dfull
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH -t 13-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie
#SBATCH --output=/home/people/22206468/slurm-ssl_pretrain_Dfull-%j.out

echo "===== JOB START ====="; date
echo "Node: $(hostname) | Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unset} | nproc: $(nproc)"

module purge
module load anaconda3
conda activate torch_v100_py310
cd ~/SSL_DATA_DISTRIBUTION

# SSL pretraining on D_full (~28.7M windows) -- shared encoder for R2 AND R3.
# Trained ONCE; loading the full manifest is memory-heavy (hence --mem=48G).
# Re-submitting this job auto-resumes.
python scripts/pretrain_ssl.py --ssl-data D_full

echo "===== JOB END ====="; date

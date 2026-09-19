#!/bin/bash -l
#SBATCH --job-name=ssl_pretrain_D0
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH -t 13-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie
#SBATCH --output=/home/people/22206468/slurm-ssl_pretrain_D0-%j.out

echo "===== JOB START ====="; date
echo "Node: $(hostname) | Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unset} | nproc: $(nproc)"

module purge
module load anaconda3
conda activate torch_v100_py310
cd ~/SSL_DATA_DISTRIBUTION

# SSL pretraining on D0 (for R1). Re-submitting this job auto-resumes.
# Set --m3-params if best_multiscale_params.json is not auto-resolved.
python scripts/pretrain_ssl.py --ssl-data D0

echo "===== JOB END ====="; date

#!/bin/bash
#SBATCH --job-name=yolov8_WH_2025
#SBATCH --partition=fit
#SBATCH --qos=fitq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "=========================================="

module load miniforge3
conda activate yolov8

echo "Python executable:"
which python

echo "Python version:"
python --version

echo "Checking GPU:"
nvidia-smi

echo "Checking YOLO/Ultralytics:"
python -c "import ultralytics; print(ultralytics.__version__)"

echo "Starting Python script..."
python -u /scratch2/bm75/mrat0006/Polytrack/src/main.py --config /scratch2/bm75/mrat0006/Polytrack/config/custom/config_WH_2025.yaml --skip-existing

echo "=========================================="
echo "Job finished on: $(date)"
echo "=========================================="
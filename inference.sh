#!/bin/bash -l

#SBATCH --time=00:30:00
#SBATCH --partition=workshop
#SBATCH --job-name=gpt-inference
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH --output=inference-%j.out
#SBATCH --error=inference-%j.err
#SBATCH --mail-user=andreferraz@gwu.edu
#SBATCH --mail-type=all

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch inference.sh \"your prompt here\" [optional inference.py args]"
    exit 1
fi

module load pixi-pytorch-gpu

python3 /home/andreferraz/distributedLLM/inference.py "$@"

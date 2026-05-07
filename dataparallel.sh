#!/bin/bash -l

# Hello! Here is a template for submitting a multi-node GPU job via SLURM on Cerberus.
# This preserves the same core resource requests as the single-node script, but launches
# PyTorch DistributedDataParallel using torchrun across multiple nodes.

# Time limit for the full job.
# Format: dd-hh:mm:ss
#SBATCH --time=02:00:00

# Cerberus partition / queue.
#SBATCH --partition=workshop

# Multi-node DDP resource request.
# Keep --nodes, --ntasks, and torchrun --nnodes aligned.
# This version runs 1 training process per node and 1 GPU per process.
#SBATCH --job-name=multinode-gpt-ddp
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=16G
#SBATCH --gres=gpu:v100:1

# Output and error logs.
#SBATCH --output=ddp-%j.out
#SBATCH --error=ddp-%j.err

# Email notifications.
#SBATCH --mail-user=andreferraz@gwu.edu
#SBATCH --mail-type=all

module load pixi-pytorch-gpu

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node="${nodes[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"
echo "Allocated nodes: ${nodes[*]}"

export LOGLEVEL=INFO

srun torchrun \
    --nnodes "$SLURM_NNODES" \
    --nproc_per_node 1 \
    --rdzv_id "$SLURM_JOB_ID" \
    --rdzv_backend c10d \
    --rdzv_endpoint "$head_node_ip:29500" \
    /home/andreferraz/distributedLLM/train_distributed.py \
    --dataset_path /home/andreferraz/wikitext_tensors.pt \
    --model_path /home/andreferraz/distributedLLM/checkpoint/model.pt \
    --metrics_path /home/andreferraz/distributedLLM/checkpoint/training_metrics.csv

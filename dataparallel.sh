#!/bin/bash -l

# Hello! Here is a template for submitting a multi-node GPU job via SLURM on Cerberus.
# This preserves the same core resource requests as the single-node script, but launches
# PyTorch DistributedDataParallel using torchrun across multiple nodes.

# Time limit for the full job.
# Format: dd-hh:mm:ss
#SBATCH --time=02:00:00

# Cerberus partition / queue.
#SBATCH --partition=workshop

#SBATCH --gres=gpu:v100:1

# Multi-node DDP resource request.
# Keep --nodes, --ntasks, and torchrun --nnodes aligned.
# This version runs 1 training process per node and 1 GPU per process.
#SBATCH --job-name=multinode-gpt-ddp
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G

# Output and error logs.
#SBATCH --error=ddp-%j.err

# Email notifications.
#SBATCH --mail-user=andreferraz@gwu.edu
#SBATCH --mail-type=all

module load pixi-pytorch-gpu

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node="${nodes[0]}"

echo "Head node: $head_node"
echo "Allocated nodes: ${nodes[*]}"

export LOGLEVEL=INFO
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=INFO

srun torchrun \
    --nnodes "$SLURM_NNODES" \
    --nproc_per_node 1 \
    --rdzv_id "$SLURM_JOB_ID" \
    --rdzv_backend c10d \
    --rdzv_endpoint "$head_node:29500" \
    /home/andreferraz/distributedLLM/train_distributed.py \
    --dataset_path /home/andreferraz/wikitext_tensors.pt \
    --model_path /home/andreferraz/distributedLLM/checkpoint/model.pt \
    --metrics_path /home/andreferraz/distributedLLM/checkpoint/training_metrics.csv

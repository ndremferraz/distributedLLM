#!/bin/bash -l

# Multi-node FSDP2 job submission for Cerberus.
# This follows the same SLURM and torchrun pattern as the working DDP script,
# but launches the FSDP2 training entrypoint instead.

# Time limit for the full job.
# Format: dd-hh:mm:ss
#SBATCH --time=02:00:00

# Cerberus partition / queue.
#SBATCH --partition=workshop

# Request one typed GPU per node.
#SBATCH --gres=gpu:v100:1

# Multi-node FSDP2 resource request.
# This version runs 1 training process per node and 1 GPU per process.
#SBATCH --job-name=multinode-gpt-fsdp2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G

# Output and error logs.
#SBATCH --output=fsdp-%j.out
#SBATCH --error=fsdp-%j.err

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun torchrun \
    --nnodes "$SLURM_NNODES" \
    --nproc_per_node 1 \
    --rdzv_id "$SLURM_JOB_ID" \
    --rdzv_backend c10d \
    --rdzv_endpoint "$head_node:29500" \
    /home/andreferraz/distributedLLM/fsdp2_train.py \
    --dataset_path /home/andreferraz/wikitext_tensors.pt \
    --model_path /home/andreferraz/distributedLLM/checkpoint/fsdp_model.pt \
    --metrics_path /home/andreferraz/distributedLLM/checkpoint/fsdp_training_metrics.csv

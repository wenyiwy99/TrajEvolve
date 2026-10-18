#!/bin/bash
# Discriminator trainer SFT training using verl FSDP SFT Trainer

set -x

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

# NCCL configuration
export NCCL_TIMEOUT="3600"
export NCCL_P2P_LEVEL="NVL"
export NCCL_IB_TIMEOUT="24"
export NCCL_NVLS_ENABLE="0"
export NCCL_MPI_PROFILE_PRIMS_ENABLE="0"
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="3600"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_SL="3"
export NCCL_CHECK_DISABLE="1"
export NCCL_P2P_DISABLE="0"
export NCCL_IB_DISABLE="0"
export NCCL_LL_THRESHOLD="16384"
export NCCL_IB_CUDA_SUPPORT="1"
export NCCL_SOCKET_IFNAME="bond1"
export UCX_NET_DEVICES="bond1"
export NCCL_IB_HCA="mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6"
export NCCL_COLLNET_ENABLE="0"
export SHARP_COLL_ENABLE_SAT="0"
export NCCL_NET_GDR_LEVEL="2"
export NCCL_IB_QPS_PER_CONNECTION="4"
export NCCL_IB_TC="160"
export NCCL_PXN_DISABLE="1"

# Path configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CREDIT_DATA_DIR="${SCRIPT_DIR}/../credit_analysis_data/alfworld"
OUTPUT_DIR="$HOME/trajevolve/discriminator_model/alfworld"
MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct

# Step 1: Data processing (ALFWorld: prompt includes action and observation)
echo "Processing data..."
python3 -u ${SCRIPT_DIR}/data_processor.py \
    --input_file ${CREDIT_DATA_DIR}/credit_analysis.json \
    --output_dir ${CREDIT_DATA_DIR} \
    --env_type alfworld

# Step 2: verl FSDP SFT training (8-GPU distributed training)
echo "Starting verl FSDP SFT training..."
torchrun --nproc_per_node=8 \
    ${SCRIPT_DIR}/trainer.py \
    data.train_files=${CREDIT_DATA_DIR}/train.parquet \
    data.val_files=${CREDIT_DATA_DIR}/train.parquet \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=4096 \
    model.partial_pretrain=$MODEL_PATH \
    model.enable_gradient_checkpointing=True \
    optim.lr=1e-5 \
    optim.warmup_steps_ratio=0.1 \
    optim.weight_decay=0.01 \
    trainer.total_epochs=3 \
    trainer.default_local_dir=${OUTPUT_DIR}

echo "Training completed. Model saved to ${OUTPUT_DIR}"

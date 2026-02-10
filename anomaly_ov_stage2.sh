#!/bin/bash
#SBATCH -J Anomaly_OV_Stage2_FT
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH -t 24:00:00
#SBATCH -o /work/foobarbaz911/anomaly-ov/logs/train_stage2_%j.out
#SBATCH -e /work/foobarbaz911/anomaly-ov/logs/train_stage2_%j.err
#SBATCH --account=mst114553
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mingfeng8886731@gmail.com

set -euo pipefail

module load singularity/3.7.1 cuda/12.4

export REPO="/home/foobarbaz911/Anomaly-OneVision"
export WORK="/work/foobarbaz911/anomaly-ov"
export IMG="${WORK}/images/anomaly-ov.sif"
export PIP_SITE="/work/foobarbaz911/pip/lib/python3.10/site-packages"
export MODEL_DIR="${WORK}/models/llava-onevision-qwen2-7b-ov"
export HF_HOME="${WORK}/hf_cache"
export HF_HUB_CACHE="${WORK}/hf_cache"
export HF_DATASETS_CACHE="${WORK}/hf_cache"
export XDG_CACHE_HOME="${WORK}/xdg_cache"

singularity exec --nv --pwd /app \
  -B "${REPO}":/app \
  -B "${WORK}":/work \
  -B /home/foobarbaz911:/home/foobarbaz911 \
  -B /work/foobarbaz911:/work2 \
  -B /work/HPC_software/LMOD/nvidia/packages/cuda-12.4:/cuda \
  "${IMG}" bash -lc "
    set -euo pipefail

    export PATH=/opt/venv/bin:\$PATH
    export CUDA_HOME=/cuda
    export LD_LIBRARY_PATH=/cuda/lib64:/cuda/lib:${LD_LIBRARY_PATH:-}
    export PYTHONPATH=/work2/pip/lib/python3.10/site-packages:${PYTHONPATH:-}
    export HF_HOME=/work/hf_cache
    export HF_HUB_CACHE=/work/hf_cache
    export HF_DATASETS_CACHE=/work/hf_cache
    export XDG_CACHE_HOME=/work/xdg_cache
    export WANDB_MODE=online
    export WANDB__SERVICE_WAIT=300
    export WANDB_DIR=/work/outputs/wandb
    mkdir -p \$WANDB_DIR

    cd /app

    /opt/venv/bin/torchrun \
      --nproc_per_node=4 \
      --nnodes=1 \
      --master_port=29511 \
      /app/llava/train/train_mem.py \
      --deepspeed /app/scripts/zero2.json \
      --model_name_or_path /work/models/llava-onevision-qwen2-7b-ov \
      --version qwen_1_5 \
      --data_path /app/data/datasets.yaml \
      --image_folder /work/data \
      --mm_tunable_parts \"mm_mlp_adapter,mm_language_model\" \
      --vision_tower google/siglip-so400m-patch14-384 \
      --output_dir /work/outputs/anomalyov_7B_stage2 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 2 \
      --bf16 True \
      --model_max_length 32768 \
      --gradient_checkpointing True \
      --report_to wandb
  "

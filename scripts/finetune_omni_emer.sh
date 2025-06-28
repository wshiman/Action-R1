#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
# Environment Variables
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-1}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16667
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=16
LOCAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
echo $GRADIENT_ACCUMULATION_STEPS

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=humanomniqwen2_siglip
export HF_HOME=/data/data2/shiman/R1-Omni/hfhome/
export HF_ENDPOINT=http://hf-mirror.com
RUN_NAME=dataplus_coldstart

OUTP_DIR=work_dirs
echo 'hello'
torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    /data/data2/shiman/R1-Omni/humanomni/train_flash_attn.py \
    --deepspeed /data/data2/shiman/R1-Omni/scripts/zero3.json \
    --model_type HumanOmni_qwen2 \
    --model_path /data/data2/shiman/R1-Omni/humanomni-0.5B \
    --vision_tower /data/data2/shiman/R1-Omni/siglip-224 \
    --mm_projector_type all_in_one_small \
    --mm_tunable_parts "mm_mlp_adapter,mm_language_model" \
    --data_path   /data/data2/shiman/R1-Omni/cold_start/cold_start++_random_updated.json \
    --data_folder / \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --num_frames 8 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/finetune_${RUN_NAME} \
    --num_train_epochs 10 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 99 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --mm_use_x_start_end True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \

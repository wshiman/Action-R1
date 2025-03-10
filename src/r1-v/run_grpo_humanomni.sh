cd src/r1-v
VIDEOLLAMA_DIR=/mnt/data/jiaxing.zjx/code/HumanOmni/
CURRENT_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$VIDEOLLAMA_DIR:/mnt/data/jiaxing.zjx/code/R1-V-Qwen/R1-V:/mnt/data/jiaxing.zjx/code/R1-V-Qwen/R1-V/src/r1-v/src/:/mnt/data/jiaxing.zjx/code/R1-V-Qwen/R1-V/src/r1-v/

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./logs/humanomni_emotion_emer_1format_withpath_withchoice.txt"
export HF_HOME=/mnt/data/jiaxing.zjx/cache/huggingface/

WANDB_MODE=offline torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo.py \
    --output_dir ./outputs/test_humanomni_emer_1format_withpath_withchoice/ \
    --model_name_or_path /mnt/data/jiaxing.zjx/code/HumanOmni/work_dirs/humanomniqwen2_siglip/finetune_HumanOmni_1B_Omni_emer_withchoice \
    --dataset_name /mnt/data/jiaxing.zjx/code/R1-V-Qwen/R1-V/leonardPKU/clevr_cogen_a_train \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-emotion \
    --save_steps 1000 \
    --save_only_model true \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
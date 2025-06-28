cd src/r1-v

export CUDA_VISIBLE_DEVICES=5,6
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./logs/humanomni_action_$(date +%Y%m%d_%H%M%S).txt" # Log path for training logs, add time stamp to the filename
export HF_HOME=R1-Omni/hfhome/
mkdir -p ./logs

export PYTHONPATH=$PYTHONPATH:/data/data2/shiman/R1-Omni/src/r1-v/src:/data/data2/shiman/R1-Omni
# 打印日志路径以确认
echo "Log path set to: $LOG_PATH"

WANDB_MODE=offline torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo.py \
    --output_dir ./outputs/refine_final_train/ \
    --model_name_or_path /data/data2/shiman/R1-Omni/src/r1-v/outputs/dataaccupdate_final/checkpoint-1600 \
    --dataset_name /data/data2/shiman/R1-Omni/data_json/refinement_train_conversation.json \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --temporal true \
    --len_control true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.0 \
    --seed 42 \
    --lr_scheduler_type linear \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --run_name Qwen2-VL-2B-GRPO-action \
    --save_steps 50 \
    --save_only_model false \
    --report_to tensorboard \
    --num_generations 16   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

import sys
sys.path.append('./')

from humanomni.train_humanomni import train

# 需要在/data/data2/shiman/R1-Omni目录下运行bash /data/data2/shiman/R1-Omni/scripts/finetune_omni_emer.sh;否则会报错module not found

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

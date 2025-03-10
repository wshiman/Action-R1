# R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning

[![ModelScope](https://img.shields.io/badge/ModelScope-HumanOmni-blue)](https://modelscope.cn/models/myroot/R1-Omni-0.5B)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-HumanOmni-yellow)](https://huggingface.co/StarJiaxing/R1-Omni-0.5B)
[![arXiv](https://img.shields.io/badge/arXiv-2503.05379-red)](https://arxiv.org/abs/2503.05379)

<div align="center">
  <img src="figures/arch.png" width="800"/>
</div>

## 📖 Introduction
**R1-Omni** is the industry’s first application of Reinforcement Learning with Verifiable Reward (RLVR) to an Omni-multimodal large language model. We focus on emotion recognition, a task where both visual and audio modalities play crucial roles, to validate the potential of combining RLVR with Omni model. Our findings reveal several key insights:
1) **Enhanced Reasoning Capability**: R1-Omni demonstrate superior reasoning abilities, enabling a clearer understanding of how visual and audio information contribute to emotion recognition.
2) **Improved Understanding Capability**: Compared to SFT, RLVR significantly boosts performance on emotion recognition tasks.
3) **Stronger Generalization Capability**: RLVR models exhibit markedly better generalization capabilities, particularly excelling in out-of-distribution scenarios.



## 📦 Model Download
We chose the open-source Omni model HumanOmni-0.5B as our base model. We have open-sourced the following: the base model HumanOmni-0.5B, the cold-start model EMER-SFT, the model MAFW-DFEW-SFT fine-tuned directly on the MAFW and DFEW training sets, and our final model R1-Omni.
<div align="center">

| **Model**              | **HuggingFace**                                                                 | **ModelScope**                                                          |
|------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| `HumanOmni-0.5B`      |  [![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://hf.co/StarJiaxing/HumanOmni-7B-Video) | [![MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/iic/HumanOmni-7B-Video) |
| `EMER-SFT`      |  [![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://hf.co/StarJiaxing/HumanOmni-7B-Audio)  | [![MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/iic/HumanOmni-7B-Audio)  |
| `MAFW-DFEW-SFT`       | [![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://hf.co/StarJiaxing/HumanOmni-7B)         | [![MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/iic/HumanOmni-7B)         |
| `R1-Omni`       | [![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://hf.co/StarJiaxing/HumanOmni-7B)         | [![MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/iic/HumanOmni-7B)         |
</div>



## 🏆 Performance

Below are the performance on emotion recognition datasets. We use symbols to indicate whether the data is **in-distribution (⬤)** or **out-of-distribution (△)**.

| Method                          | DFEW (WAR) ⬤ | DFEW (UAR) ⬤ | MAFW (WAR) ⬤ | MAFW (UAR) ⬤ | RAVDESS (WAR) △ | RAVDESS (UAR) △ |
|----------------------------------|---------------|---------------|---------------|---------------|------------------|------------------|
| HumanOmni-0.5B                   | 22.64         | 19.44         | 20.18         | 13.52         | 7.33             | 9.38             |
| EMER-SFT                         | 35.98         | 43.24         | 25.00         | 32.60         | 29.00            | 27.19            |
| MAFW-DFEW-SFT                    | 59.61         | 71.25         | 38.89         | 52.55         | 29.33            | 30.75            |
| R1-Omni                          | 63.41         | 74.43         | 41.62         | 54.31         | 43.00            | 44.69            |

### Legend
- **⬤**: Indicates **in-distribution data** (DFEW and MAFW).
- **△**: Indicates **out-of-distribution data** (RAVDESS).




## 📺  Demo
https://github.com/user-attachments/assets/ecf07e98-f448-4cb6-a124-6e2a2885e581
```
User:What's the major emotion?
HumanOmni:angry.

User:What's the major facial expression?
HumanOmni:frown.

User:Desribe the woman.
HumanOmni:The woman has long brown hair and is wearing a pink top with a dark cardigan.

User:Please describe the major action in detail.
HumanOmni:The woman is frowning and shaking her head, indicating that she is angry. She is also gesturing with her hands as if she is explaining something or emphasizing a point.

User:What did the woman say?
HumanOmni:The woman said "Of course we object! I object, Steve!"
```

https://github.com/user-attachments/assets/0f0d9c5c-6e1d-473f-8c6b-819ebdc8aae8
```
User:What's the major emotion?
HumanOmni:happy.

User:What's the major facial expression?
HumanOmni:smile.

User:Desribe the woman in detail.
HumanOmni:The woman in the video has blonde hair, is wearing a black top, and is smiling.

User:What did the woman say?
HumanOmni:The woman said "Dogs are sitting by the door."
```

[https://humanmllm.github.io/HumanOmni/chinese_video.mp4](https://github.com/user-attachments/assets/65b49d37-7213-452d-a492-cfddbbc8b41d)
```
User:描述一下视频中男子的外貌穿着.
HumanOmni:视频中的男性有短的黑发，配戴圆框眼镜。他穿了一件棕色的毛大衣，内里是白色的.

User:视频中男子说了什么内容？
HumanOmni:男子说他穿什么颜色的衣服，现在处于什么环境，今天的天气怎么样.

User:视频中男子有什么情绪吗？
HumanOmni:没有情绪。
```

## 🛠️ Environment Setup

To set up the recommended environment for HumanOmni, follow these instructions:

### Recommended Environment
- **Python**: >=3.10
- **CUDA**: >=12.1
- **PyTorch**: >=2.2 (with CUDA support)
- **Transformers**: >=4.45
- **Accelerate**: >=0.30.1

Or you can quickly set up the environment as follows:

```
git clone https://github.com/HumanMLLM/HumanOmni
cd HumanOmni
conda create -n humanOmni python=3.10 -y
conda activate humanOmni
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
## 🧠 Training on Custom Dataset
### Data Preparation
An example json file of the training data:
```
[
    {
        "video": "human/DFEW/videos/1.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video>\n<audio>\nAs an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?\nfear ,angry ,surprise ,happy ,neutral ,sad ,disgust"
            },
            {
                "from": "gpt",
                "value": "sad"
            }
        ],
    },
    {
        "video": "human/DFEW/videos/1.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video>\n<audio>\nAs an emotional recognition expert, in the video, when the characters display their emotions, which predominant feeling is most clearly expressed?\nfear ,disgust ,happy ,sad ,surprise"
            },
            {
                "from": "gpt",
                "value": "sad"
            }
        ],
    },
  ...
]
```

### Multi-Modal SFT
- Download the required weights: (1) [HumanOmni-7B-Video](https://modelscope.cn/models/iic/HumanOmni-7B-Video) (2) [HumanOmni-7B-Audio](https://modelscope.cn/models/iic/HumanOmni-7B-Audio)
- scripts/train/finetune_humanomni.sh Loading the weights and the prepared dataset.
- bash scripts/train/finetune_humanomni.sh

## 🔍 Inference
We provide inference.py for singe video inference. 
 - video + audio 
```
python inference.py --modal video_audio \
  --model_path ./HumanOmni_7B \
  --video_path video.mp4 \
  --instruct "Describe this video."
```
 - only video 
```
python inference.py --modal video \
  --model_path ./HumanOmni_7B \
  --video_path video.mp4 \
  --instruct "Describe this video."
```
- only audio
```
python inference.py --modal audio \
  --model_path ./HumanOmni_7B \
  --video_path video.mp4 \
  --instruct "Describe this video."
```

## 🤝 Related Work
- [LLaVA-Octopus: Unlocking Instruction-Driven Adaptive Projector Fusion for Video Understanding](https://arxiv.org/abs/2501.05067)
- [Omni-Emotion: Extending Video MLLM with Detailed Face and Audio Modeling for Multimodal Emotion Analysis](https://arxiv.org/abs/2501.09502)
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)

## 📚 Citation
If you find our work helpful, feel free to give us a cite.
```
@article{zhao2025humanomni,
  title={HumanOmni: A Large Vision-Speech Language Model for Human-Centric Video Understanding},
  author={Zhao, Jiaxing and Yang, Qize and Peng, Yixing and Bai, Detao and Yao, Shimin and Sun, Boyuan and Chen, Xiang and Fu, Shenghao and Wei, Xihan and Bo, Liefeng and others},
  journal={arXiv preprint arXiv:2501.15111},
  year={2025}
}
```

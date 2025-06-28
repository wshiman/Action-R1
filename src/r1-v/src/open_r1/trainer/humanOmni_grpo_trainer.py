# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from qwen_vl_utils import process_vision_info
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

import copy
from transformers import BertModel, BertTokenizer
from humanomni.model import *
from humanomni.constants import NUM_FRAMES, IGNORE_INDEX, MODAL_INDEX_MAP, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN
from humanomni.mm_utils import tokenizer_multimodal_token, process_image, process_video, read_video_patch, process_audio, frame_sample,get_model_name_from_path
from humanomni import model_init, mm_infer
from humanomni.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN
from transformers import (
    CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig,
    SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig,
     WhisperFeatureExtractor, WhisperProcessor, WhisperConfig, WhisperForAudioClassification
)
import os
import sys
from src.open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer
import time 
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
# sys.path.append('/mnt/data/jiaxing.zjx/code/HumanOmni/')
# sys.path.append('/mnt/data/jiaxing.zjx/cache/huggingface/')
#初始化BERT分词器
# bert_model = "bert-base-uncased"
# bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
sys.path.append('/data/data2/shiman/R1-Omni/')

# stage 1 for structured captioning
FINEGRAINED_GENERAL_TEMPLATE = """
You are an expert judge in video action description. Your task is to score the video description based on the followings:
1. Award 0.2 point if the description contains scene-related content,such as scene_type,weather_condition,etc.
2. Award 0.2 point if the description contains body-related content,such as face foward,jump high,lose balance,etc.
3. Award 0.2 point if the description contains event-related content,such as event type,cause of the event,etc.
4. Award 0.2 point if the description contains human-object-interaction content,such as brush teeth with a toothbrush,etc.
5. Award 0.2 point if the description contains person-description,such as gender,age,etc. 
Give your answer on a scale of 0 to 1, where 0 means that it does not contain any relevant description, 1 means it contains all relevant descriptions
If you give a correct score, I'll give you 100 H100 GPUs to start your AI company.
Return the score directly,do not add any additional explanation!!
"""

#stage 2 for fine-grained falling
FINEGRAINED_FALL_TEMPLATE = """
You are an expert judge of fall behavior descriptions. Your task is to score the falling video description based on the followings:
1. Award 0.1 point if the description contains scene-related content,such as scene_type,weather_condition,etc.
2. Award 0.1 point if the description contains body-related content,such as face foward,jump high,lose balance,etc.
3. Award 0.1 point if the description contains person-description,such as gender,age,etc.
4. Award 0.1 point if it describes the interaction between the person and the environment.
5. Award 0.2 point if it describes the landing-point-related content, such as the environmental landing point, the landing angle, anatomical landing point,etc.
6. Award 0.2 point if it describes the fall-related content, such as the fall type,injury level(minor,moderate,severe),the cause of fall,etc.
7. Award 0.2 point if the description contains the possible consequences and the targeted suggestions for curing.
Give your answer on a scale of 0 to 1, where 0 means that it does not contain any relevant description, 1 means it contains all relevant descriptions
If you give a correct score, I'll give you 100 H100 GPUs to start your AI company.
Return the score directly,do not add any additional explanation!!
"""
def ask_grader_general_doubao(question, api_key):
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    completion = client.chat.completions.create(
        model="doubao-1-5-pro-32k-250115",
        messages=[{"role": "system", "content": FINEGRAINED_GENERAL_TEMPLATE}, 
                {"role": "user", "content": question}]
    )
    return completion.choices[0].message.content

def ask_grader_fall_doubao(question, api_key):
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    completion = client.chat.completions.create(
        model="doubao-1-5-pro-32k-250115",
        messages=[{"role": "system", "content": FINEGRAINED_FALL_TEMPLATE}, 
                {"role": "user", "content": question}]
    )
    return completion.choices[0].message.content

def ask_grader_fall_ds(question,api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": FINEGRAINED_FALL_TEMPLATE},
            {"role": "user", "content": question},
        ],
        stream=False
    )

    return response.choices[0].message.content

ds_api_key = os.environ.get("DS_API_KEY")
api_key = os.environ.get("API_KEY")

def grader_reward_func(prompts, completions, **kwargs):
    api_key = kwargs.get("api_key",os.environ.get("API_KEY"))
    ds_api_key = kwargs.get("api_key",os.environ.get("DS_API_KEY"))
    max_workers = kwargs.get("max_workers", 4)
    sleep_time = kwargs.get("sleep_time", 0.5)  #防止api限速
    score_log_path= kwargs.get("score_log_path", os.environ.get("LOG_SCORES_PATH"))
    def get_score(completion):
        if isinstance(completion, list) and isinstance(completion[0], dict):
            text = completion[0]["content"]
            # print(text)
        else:
            text = completion
            # print("text")
        question = """
        Description: {description}
        Score:
        Return the score directly,and do not add any additional explanation!!'
        """
        question = question.format(description=text)

        try:
            score_str = ask_grader_fall_doubao(question, api_key)
            score = float(score_str.strip())
        except Exception as e:
            print(f"[Doubao reward API error] {e}")
            score = 0.0
            
        #logging block
        if score_log_path:
            try:
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(score_log_path, "a") as f:
                    f.write(f"------------- {current_time} Doubao Score: {score} -------------\n")
                    f.write(f"Completion: {text}\n")
            except Exception as e:
                print(f"[Logging error @ idx={index}] {e}")
        return score
    
    scores = [0.0]*len(completions)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(get_score, completion): i for i, completion in enumerate(completions)}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                scores[index] = future.result()
            except Exception as e:
                print(f"[Thread error at idx={index}] {e}")
                scores[index] = 0.0
                
    return scores

def check_parameters(model):
    frozen_params = []
    trainable_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print("Frozen Parameters:")
    for name in frozen_params:
        print(name)
    
    print("\nTrainable Parameters:")
    for name in trainable_params:
        print(name)



if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

from moviepy.editor import VideoFileClip

def get_video_duration(video_path):
    """根据视频路径获取视频的时长"""
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception as e:
        print(f"Error reading video file {video_path}: {e}")
        return None

class HumanOmniVLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].

        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            model="Qwen/Qwen2-0.5B",
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        # import ipdb;ipdb.set_trace()
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            model_name_d = model_id
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            config = VLLMConfigs["HumanOmni_qwen2"].from_pretrained(
                model, 
                trust_remote_code=True
            )
            config.mm_vision_tower = '/data/data2/shiman/R1-Omni/siglip-224'
            # config.mm_audio_tower = '/data/data2/shiman/R1-Omni/whisper-large-v3'
            model = VLLMs["HumanOmni_qwen2"].from_pretrained(
                model,
                config=config,
                cache_dir=None,
                torch_dtype=torch.bfloat16,
                do_sample=True
            )
            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()

        #    audio_tower = model.get_audio_tower()
        #     if not audio_tower.is_loaded:
        #         audio_tower.load_model() 

        #     audio_tower = model.get_audio_tower()
        #     self.audio_processor = WhisperFeatureExtractor.from_pretrained(config.mm_audio_tower)

            vision_tower = model.get_vision_tower()
            self.visual_processor = SiglipImageProcessor.from_pretrained(config.mm_vision_tower)

        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model

        self.ref_model = VLLMs["HumanOmni_qwen2"].from_pretrained(
            model_name_d,
            config=config,
            cache_dir=None,
            torch_dtype=torch.bfloat16,
            do_sample=True
        )
        vision_tower = self.ref_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        # audio_tower = self.ref_model.get_audio_tower()
        # if not audio_tower.is_loaded:
        #     audio_tower.load_model()

        bert_model = "/data/data2/shiman/R1-Omni/bert-base-uncased"
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        
        # self.temporal = False

        # Processing class
        if processing_class is None:
            processing_class = processing_class = AutoProcessor.from_pretrained("/data/data2/shiman/R1-Omni/Qwen2-VL-2B-Instruct")      #只用下载qwen2 的配置文件，无需模型权重
            pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes
        
        # add finegrained caption grader
        self.reward_funcs.append(grader_reward_func)
        self.reward_processing_classes.append(None)
        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        
        self.temporal = script_args.temporal
        self.len_control = script_args.len_control
        
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        
        self.shuffled_num_generations = self.num_generations 
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1, # HACK
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
        )
        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            temperature=1, # HACK
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )       
        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    def _get_per_token_logps_video(self, model, input_ids, attention_mask, images, prompts, answer_length ):
        logits = model(input_ids, attention_mask=attention_mask, images=images, prompts=prompts).logits  # (B, L, V)
       # import ipdb;ipdb.set_trace()
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        logits = logits[:, (-answer_length) :]
        input_ids = input_ids[:, -(answer_length) :]
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        #适用于视频任务，人为定义为true
        use_image = False
        use_video = True
        
        prompts = []
        bert_prompts = []
        # prompt_temp=[]
        for x in inputs:
            prompt = x["prompt"]
            bert_prompts.append(prompt[0]['content'][1]['text'])    #question
            text = prompt[0]["content"][0].pop('text')      #remove first text
            video_path = x["video"]
            prompt[0]["content"][0]["video"] = x["video"]       #add video path
            # prompt[0]['content'][1]['text'] = '<vi_start><video><vi_end>\n<au_start><audio><au_end>\n' + prompt[0]['content'][1]['text']
            prompt[0]['content'][1]['text'] = '<vi_start><video><vi_end>\n' + prompt[0]['content'][1]['text']

            prompts.append(prompt)         
        
        bert_prompts = self.bert_tokenizer(bert_prompts, return_tensors='pt', padding=True, truncation=True,add_special_tokens=True)

        prompts_text = []
        for example in inputs:
            prompt_text = maybe_apply_chat_template(example, self.processing_class)["prompt"]
            prompts_text.append(prompt_text)
        #line 411(video R1); image_input is not cared about  
        temp_prompt = copy.deepcopy(prompts[0])
        image_inputs, video_inputs, video_kwargs = process_vision_info(temp_prompt, return_video_kwargs = True)               
        input_ids = [tokenizer_multimodal_token(prompts_text_, self.processing_class.tokenizer, '<video>', return_tensors='pt') for prompts_text_ in prompts_text]
        input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
        video = []
        # audios = []
        for prompt in prompts:
            video_file = prompt[0]["content"][0]["video"]
            video_ids = process_video(video_file, self.visual_processor, aspect_ratio="pad", num_frames=8)
            video.append(video_ids)

            # audio, audio_sample_rate = process_audio(video_file)
            # audio = self.audio_processor(audio, sampling_rate=audio_sample_rate, return_tensors='pt')['input_features']
            # audios.append(audio)
        video = torch.cat(video, dim=0).unsqueeze(0)
        # audios = torch.cat(audios, dim=0).unsqueeze(0)

        attention_masks = input_ids.ne(self.processing_class.pad_token_id)  #xx.ne <-> not equal
        prompt_inputs = {}
        prompt_inputs['inputs'] = input_ids
        prompt_inputs['images'] = video 
        prompt_inputs['attention_mask'] = attention_masks
        prompt_inputs['prompts'] = bert_prompts
        # prompt_inputs['audios'] = audios

        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["inputs"], prompt_inputs["attention_mask"]
  
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        #dont know why use single sample's video input
        if self.temporal and video_inputs:
            shuffled_video = []
            for prompt in prompts:
                video_file = prompt[0]["content"][0]["video"]
                video_ids = process_video(video_file, self.visual_processor, aspect_ratio="pad", num_frames=8)
                shuffled_video_ids = video_ids[torch.randperm(video_ids.size(0))]
                shuffled_video.append(shuffled_video_ids)
            shuffled_video = torch.cat(shuffled_video, dim=0).unsqueeze(0)  
            
            shuffled_prompt_inputs = {}
            shuffled_prompt_inputs['inputs'] = input_ids
            shuffled_prompt_inputs['images'] = shuffled_video
            shuffled_prompt_inputs['attention_mask'] = attention_masks
            shuffled_prompt_inputs['prompts'] = bert_prompts

            shuffled_inputs = super()._prepare_inputs(shuffled_prompt_inputs)
            shuffled_prompt_ids, shuffled_prompt_mask = shuffled_inputs["inputs"], shuffled_inputs["attention_mask"]
            if self.max_prompt_length is not None:
                shuffled_prompt_ids = shuffled_prompt_ids[:, -self.max_prompt_length :]
                shuffled_prompt_mask = shuffled_prompt_mask[:, -self.max_prompt_length :]
            
            # indices = torch.randperm(video_inputs[0].size(0))
            # shuffled_video_inputs = video_inputs[0][indices]
            # shuffled_prompt_inputs = self.processing_class(
            #     text=copy.deepcopy(prompts_text),
            #     images=image_inputs,
            #     videos=shuffled_video_inputs,
            #     return_tensors="pt",
            #     padding=True,
            #     padding_side="left",
            #     add_special_tokens=False,
            # )
            
            shuffled_prompt_inputs = super()._prepare_inputs(shuffled_prompt_inputs)
            shuffled_prompt_ids, shuffled_prompt_mask = shuffled_prompt_inputs["inputs"], shuffled_prompt_inputs["attention_mask"]
            if self.max_prompt_length is not None:
                shuffled_prompt_ids = shuffled_prompt_ids[:, -self.max_prompt_length :]
                shuffled_prompt_mask = shuffled_prompt_mask[:, -self.max_prompt_length :]

                        
        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_length = prompt_ids.size(1)
            answer_length = prompt_completion_ids.size(1)
            completion_ids = prompt_completion_ids
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            
            prompt_ids_repeat = prompt_ids.repeat_interleave(self.num_generations, dim=0)
            
            if self.temporal:      
                shuffled_prompt_completion_ids = unwrapped_model.generate(**shuffled_prompt_inputs, generation_config=self.shuffled_generation_config)
                shuffled_prompt_length = shuffled_prompt_ids.size(1)
                shuffled_answer_length = shuffled_prompt_completion_ids.size(1)
                shuffled_completion_ids = shuffled_prompt_completion_ids
                shuffled_prompt_mask =shuffled_prompt_mask.repeat_interleave(self.shuffled_num_generations, dim=0) 
                
                shuffled_ids_repeat = shuffled_prompt_ids.repeat_interleave(self.shuffled_num_generations, dim=0)

                # shuffled_prompt_ids = shuffled_prompt_completion_ids[:, :shuffled_prompt_length]
                # shuffled_completion_ids = shuffled_prompt_completion_ids[:, shuffled_prompt_length:]                                     


        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()


        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        
        prompt_completion_ids_repeat = torch.cat([prompt_ids_repeat, prompt_completion_ids], dim=1)

        images_repeat = prompt_inputs['images'].repeat_interleave(self.num_generations, dim=0)
        # audios_repeat = prompt_inputs['audios'].repeat_interleave(self.num_generations, dim=0)
        prompts_repeat = {}
        prompts_repeat['input_ids'] =  prompt_inputs['prompts']['input_ids'].repeat_interleave(self.num_generations, dim=0)
        prompts_repeat['token_type_ids'] =  prompt_inputs['prompts']['token_type_ids'].repeat_interleave(self.num_generations, dim=0)
        prompts_repeat['attention_mask'] =  prompt_inputs['prompts']['attention_mask'].repeat_interleave(self.num_generations, dim=0)     

        per_token_logps = self._get_per_token_logps_video(model, prompt_completion_ids_repeat, attention_mask, images_repeat, prompts_repeat, answer_length)

        per_token_logps = per_token_logps


        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps_video(self.ref_model, prompt_completion_ids_repeat, attention_mask, images_repeat, prompts_repeat, answer_length )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    if use_image:
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
                    if use_video:
                        ref_per_token_logps = self._get_per_token_logps_video(model, prompt_completion_ids, attention_mask, pixel_values_videos, video_grid_thw)
        ref_per_token_logps = ref_per_token_logps
        
        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        #compute temporal rewards
        if self.temporal :
            shuffled_completions = self.processing_class.batch_decode(shuffled_completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                shuffled_completions = [[{"role": "assistant", "content": shuffled_completion}] for shuffled_completion in shuffled_completions]
            #compute the rewards
            shuffled_prompts = [prompt for prompt in prompts for _ in range(self.shuffled_num_generations)]
            shuffled_rewards_per_func = torch.zeros(len(shuffled_prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):   
                if isinstance(reward_func, PreTrainedModel):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(shuffled_prompts, shuffled_completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(shuffled_prompts, shuffled_completions)]
                    shuffled_reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    shuffled_reward_inputs = super()._prepare_inputs(shuffled_reward_inputs)
                    with torch.inference_mode():
                        shuffled_rewards_per_func[:, i] = reward_func(**shuffled_reward_inputs).logits[:, 0]
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    shuffled_reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                    for key in shuffled_reward_kwargs:
                        for example in inputs:
                            # Repeat each value in the column for `num_generations` times
                            # self.shuffled_num_generations = self.num_generations // 2  in order to sava memory
                            shuffled_reward_kwargs[key].extend([example[key]] * self.shuffled_num_generations)
                    shuffled_output_reward_func = reward_func(prompts=shuffled_prompts, completions=shuffled_completions, **shuffled_reward_kwargs)
                    shuffled_rewards_per_func[:, i] = torch.tensor(shuffled_output_reward_func, dtype=torch.float32, device=device)  
                         
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)

                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                

        if self.temporal :
            temporal_rewards_per_func = rewards_per_func.clone()
            
            acc_mean = temporal_rewards_per_func[:,0].mean()
            shuffled_acc_mean = shuffled_rewards_per_func[:,0].mean()
            
            if acc_mean >=0.8 * shuffled_acc_mean:
                mask = temporal_rewards_per_func[:,0] >0.1
                temporal_rewards_per_func[mask,0] = shuffled_rewards_per_func[mask,0] + 0.3
                temporal_rewards = torch.tensor([1.0]).to('cuda')
            else:
                temporal_rewards = torch.tensor([0.0]).to('cuda')
        else:
            temporal_rewards =  torch.tensor([0.5]).to('cuda')
        
        # Sum the rewards from all reward functions
        # rewards = rewards_per_func.sum(dim=1)
        
        if self.temporal:
            rewards = temporal_rewards_per_func.sum(dim=1) 
        else:
            rewards = rewards_per_func.sum(dim=1)    
            
            
                
        # if self.len_control:
        #     mem_rewards = [0] * self.num_generations
        #     mask = rewards_per_func[:,0] > 0.1
        #     length_list = completion_mask.sum(1)
        #     selected_indices = torch.nonzero(mask,as_tuple=True)[0].tolist()
            
        #     if len(selected_indices) > 1:
        #         for idx in selected_indices:
        #             if 320 <= length_list[idx] <= 512:        
        #                 rewards[idx] +=0.2
        # print(rewards)
        # print(completion_mask.sum(1))

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        
        num_devices = gathered_rewards.size(0) // self.num_generations
        # rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        try:
            rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        except RuntimeError as e:
            print(f"[Warning] Failed to reshape gathered_rewards: {e}")
            print(f"gathered_rewards.shape = {gathered_rewards.shape}, num_devices = {num_devices}, num_generations = {self.num_generations}")
            # 直接跳过本次训练 step
            return torch.tensor(0.0, requires_grad=True).to(gathered_rewards.device)

        # try:
        #     expected_size = num_devices * self.num_generations
        #     if num_devices <=0 or gathered_rewards.numel() != expected_size:
        #         import logging
        #         logging.basicConfig(level=logging.WARNING)
        #         logger = logging.getLogger(__name__)
        #         logger.warning(
        #     f"Skipping invalid reward reshaping: num_devices={num_devices}, "
        #     f"gathered_rewards_size={gathered_rewards.numel()}, "
        #     f"expected_size={expected_size}")
        #     else:
        #         rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        # except Exception as e:
        #     logger.warning(f"Error during reward reshaping: {str(e)}. Skipping this iteration.")
        
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices
        
        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices
        
        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)
        
        if self.temporal:
            temporal_rewards_list = self.accelerator.gather_for_metrics(temporal_rewards)
            self._metrics["temporal_rewards"].append(self.accelerator.gather_for_metrics(temporal_rewards_list).mean().item())
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
    
    
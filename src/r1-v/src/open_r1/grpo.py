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
import logging
import os
import pathlib
import re
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, Features,Sequence,Value
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, HumanOmniVLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from openai import OpenAI


import os
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
# deepseek api
# def ask_api(question,api_key):
#     client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant"},
#             {"role": "user", "content": question},
#         ],
#         stream=False
#     )

#     return response.choices[0].message.content

# doubao api
def ask_api_doubao(question, api_key):
    client = OpenAI(
    api_key=api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    completion = client.chat.completions.create(
        model="doubao-1-5-pro-32k-250115",
        messages=[
            {"role": "user", "content": question},
        ]
    )
    return completion.choices[0].message.content


api_key = os.getenv("API_KEY")  

def call_synonym_api(ground_truth, student_answer):
    question = f"Whether {ground_truth} and {student_answer} have the similar meaning or express the similar action, you only have to answer yes or no"
    return ask_api_doubao(question=question, api_key=api_key).strip().lower()

def process_remaining(item):
    video, content, sol = item
    reward = 0.0
    try:
        # 提取answer标签内的内容
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

        content_match = re.search(r'<answer>(.*?)</answer>', content)
        student_answer = content_match.group(1).strip() if content_match else content.strip()

        # 调用api匹配
        if student_answer == ground_truth:
            reward = 1.0
        else:
            if call_synonym_api(ground_truth, student_answer) == "yes":
                reward = 1.0
    except Exception as e:
        print(f"Error in matching/synonym checking: {e}")

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        with open(log_path, "a") as f:
            f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
            f.write(f"Video: {video}\n")
            f.write(f"Content: {content}\n")
            f.write(f"Solution: {sol}\n")
    return reward


def accuracy_reward(completions, solution, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    contents = [completion[0]["content"] for completion in completions]
    videos = kwargs.get("video", "")
    
    rewards = []
    remaining_items = []
    pre_rewards = []

    for idx, (video, content, sol) in enumerate(zip(videos, contents, solution)):
        # 如果格式不匹配，直接给 reward -1.0
        if not re.fullmatch(pattern, content, re.DOTALL):
            rewards.append(-1.0)
            continue
        
        # 尝试通过 parse 和 verify 判断
        reward = 0.0
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass

        if reward == 1.0:
            rewards.append(1.0)
        else:
            rewards.append(None)
            remaining_items.append((idx, (video, content, sol)))  # 保留索引用于之后填回

    # 对需要进一步判断的项调用 process_remaining
    if remaining_items:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda item: process_remaining(item[1]), remaining_items))
        for (idx, _), r in zip(remaining_items, results):
            rewards[idx] = r

    return rewards

    # return 0  # delete the accuracy reward function


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def format_reward_2(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

from datasets import Dataset, DatasetDict

def load_video_dataset(json_path):
    # 读取json文件
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # 准备转换后的数据列表
    transformed_data = {
        'video': [],
        'problem': [],
        'solution': []
    }
    
    # 遍历json数据并转换
    for entry in data:
        video_path = entry['video']
        problem = None  # 初始化问题变量
        for conversation in entry['conversations']:
            if conversation['from'] == 'human':
            #  problem = conversation['value'].replace('<video>\n<audio>\n', '')
                problem = "As an action recognition expert; throughout the video, which action conveyed by the characters is the most obvious to you?"

            elif conversation['from'] == 'gpt' and problem is not None:
                solution = f"<answer> {conversation['value']} </answer>"
                # 添加到transformed_data
                transformed_data['video'].append(video_path)
                transformed_data['problem'].append(problem)
                transformed_data['solution'].append(solution)

    # 创建dataset
    dataset = Dataset.from_dict(transformed_data)
    dataset_dict = DatasetDict({'train': dataset})
    
    return dataset_dict


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    json_file_path = script_args.dataset_name
    dataset = load_video_dataset(json_file_path)
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # def make_conversation_image(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": example["problem"]},
    #                 ],
    #             },
    #         ],
    #     }

    # QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

    # QUESTION_TEMPLATE = "{Question}\nOutput the thinking process in <think> </think> and final action in <answer> </answer> tags."
    QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )
    
    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])
    elif "video" in dataset[script_args.dataset_train_split].features:
        print("has video in dataset")
        dataset = dataset.map(make_conversation_video)
    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
# trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    trainer_cls = HumanOmniVLGRPOTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    # trainer.train()
    # checkpoint_path = "/data/data2/shiman/R1-Omni/src/r1-v/outputs/checkpoint_grpoinit"

    # if os.path.exists(checkpoint_path):
    #     print(f"Resuming training from checkpoint at {checkpoint_path}")
    #     trainer.train(resume_from_checkpoint=checkpoint_path)
    # else:
    #     print("Starting training from scratch")
    #     trainer.train()
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print(f"Resuming training from checkpoint at {training_args.output_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Starting training from scratch")
        trainer.train()

    
    
    
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

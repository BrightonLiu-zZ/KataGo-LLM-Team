import os
import re
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig

# ================= 配置区域 =================
# 你的数据集路径
DATASET_PATH = "data/training_ready_data_shuffled.jsonl" 
# 模型路径 (首次运行会自动下载)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# 输出目录
OUTPUT_DIR = "runs/Qwen2.5-7B-GRPO-Go-v1"

SYSTEM_PROMPT = (
    "You are a world-class 9x9 Go expert. "
    "Before answering, strictly perform chain-of-thought reasoning within <think>...</think> tags. "
    "Analyze the board geometry, liberties, and safety of groups. "
    "Finally, output the move and a strategic explanation.\n\n"
    "Format Example:\n"
    "<think>\nThe black group on top needs eyespace. D7 is vital for defense...\n</think>\n"
    "MOVE: D7\n"
    "EXPLAIN: Expands eye space and threatens to cut.\n\n"
    "Now, analyze the board below:"
)

# ================= 辅助函数 =================

def extract_move(text: str) -> Optional[str]:
    """使用正则提取 MOVE: 后的坐标"""
    # 匹配 "MOVE: C4" 或 "MOVE:C4" 或 "MOVE: c4"，忽略大小写
    match = re.search(r"MOVE:\s*([A-HJ-T][1-9])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def check_format(text: str) -> bool:
    """检查是否符合 <think>...MOVE...EXPLAIN 格式"""
    if "<think>" not in text or "</think>" not in text:
        return False
    if "MOVE:" not in text:
        return False
    if "EXPLAIN:" not in text:
        return False
    return True

# ================= 奖励函数 (Reward Functions) =================
# TRL 会自动把数据集里的列作为参数传进来 (例如 katago_all)

def format_reward_func(completions, **kwargs) -> List[float]:
    """奖励 1: 格式检查"""
    rewards = []
    for content in completions:
        # 如果格式完美，给 +0.5 的小奖励；如果格式崩了，给 -1.0 惩罚
        if check_format(content):
            rewards.append(0.5)
        else:
            rewards.append(-1.0)
    return rewards

def katago_outcome_reward_func(prompts, completions, katago_all, **kwargs) -> List[float]:
    rewards = []
    for content, k_data in zip(completions, katago_all):
        move = extract_move(content) # 提取出的坐标，例如 "C4"
        
        # 1. 格式错误/没找到坐标 -> 重罚
        if move is None:
            rewards.append(-1.0)
            continue
            
        # 确保 k_data 解析正确
        if isinstance(k_data, str):
            try: k_data = json.loads(k_data)
            except: k_data = {}
        
        # 2. 命中 Top 10 -> 奖励
        if move in k_data:
            winrate = float(k_data[move])
            score = (winrate - 0.5) * 2.0 
            rewards.append(score)
        
        # 3. 没命中 Top 10，但是个合法的围棋坐标 -> 轻罚 (关键修改!)
        # 我们用正则判断它是不是像个坐标 (A-T + 1-9)
        elif re.match(r"^[A-HJ-T][1-9]$", move): 
            # 给一个比 -1.0 高，但比任何胜率分都低的分数
            # 假设最差的胜率是 0 (对应 -1.0)，我们给 -0.5 其实比输棋还好？
            # 不，逻辑应该是：不在 Top 10 的棋，胜率通常极低。
            # 我们可以给一个固定的“遗憾分”，比如 -0.8
            # 只要比 -1.0 (格式错误) 高，模型就会倾向于输出坐标。
            rewards.append(-0.8) 
            
        # 4. 提取出了东西但不是坐标 (比如 MOVE: Hello) -> 重罚
        else:
            rewards.append(-1.0)
            
    return rewards

# ================= 主训练逻辑 =================

def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # 1. 数据预处理：把 System Prompt 加进去，构建符合 Qwen 格式的输入
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        # TRL 的 GRPOTrainer 接受 'prompt' 字段
        # 我们在这里把 System + User 拼好，让模型只负责生成 Assistant 的回复
        formatted_prompts = []
        for user_p in examples["user_prompt"]:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_p}
            ]
            # apply_chat_template 会生成 <|im_start|>system...<|im_start|>assistant
            # add_generation_prompt=True 意味着最后会留一个口子给模型生成
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(prompt_text)
        return {"prompt": formatted_prompts}

    dataset = dataset.map(preprocess_function, batched=True)

    # 2. 配置 LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
    )

    # 3. 配置 GRPO
    # RTX 6000 Ada 显存很大，我们可以直接用 bfloat16 加载模型，不量化，性能最好
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        bf16=True, # 开启 bfloat16 加速
        per_device_train_batch_size=1, # 实际 batch = 1 * num_generations
        gradient_accumulation_steps=4, # 累计梯度
        num_generations=8,             # Group Size = 8
        max_completion_length=512,     # 生成长度限制
        max_prompt_length=1024,
        save_steps=100,
        max_steps=500, # Start Small: 先跑 500 步看看效果 (约几小时)
        report_to="tensorboard", # 或者 "wandb" 如果你有账号
        use_vllm=False, # 如果安装了 vllm 可以设为 True 加速生成，否则 False
    )

    # 4. 开始训练
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[format_reward_func, katago_outcome_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Training finished. Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
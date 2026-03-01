import json
import random
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ================= 1. 配置区域 =================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH = "/mnt/c/git_repo/KataGo-LLM-Team/data/training_data_augmented_shuffled_short.jsonl" # 你的完整数据集路径

SYSTEM_PROMPT = (
    "You are a professional 9x9 Go player. "
    "Your goal is to find the best move in the current position.\n"
    "1. FIRST, think silently about the board status, liberties, and territory in <think> tags.\n"
    "2. THEN, output your move strictly in 'MOVE: XY' format (e.g., MOVE: C4 or MOVE: pass).\n"
    "3. FINALLY, provide a short explanation starting with 'EXPLAIN:'\n"
    "\n"
    "IMPORTANT RULES:\n"
    "- DO NOT DRAW THE BOARD VISUALLY.\n"  
    "- DO NOT output ASCII art.\n"        
    "- Focus ONLY on text analysis inside <think> tags.\n"
)

# ================= 2. 辅助函数 =================
def extract_move(text: str) -> str:
    match = re.search(r"MOVE:.*?([A-HJ-T][1-9][0-9]?|pass)", text, re.IGNORECASE)
    return match.group(1).upper() if match else "N/A"

def extract_think_content(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else "N/A"

def extract_explanation(text: str) -> str:
    match = re.search(r"EXPLAIN:(.*)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else "N/A"

# ================= 3. 主程序 =================
def main():
    print("🚀 正在以 4-bit 量化加载模型 (专为 8GB VRAM 优化)...")
    
    # 核心：4-bit 量化配置，完美适配 RTX 4060
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ 模型加载完成！\n")

    # 读取数据集
    print(f"📖 正在读取数据集: {DATASET_PATH}")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 交互循环
    while True:
        input("\n🎯 按 [Enter] 随机抽取一个局面进行测试，或输入 [Ctrl+C] 退出...")
        
        # 随机抽取一条数据
        sample = json.loads(random.choice(lines))
        user_prompt = sample.get("user_prompt", "")
        katago_all = sample.get("katago_all", {})
        katago_best_wr = sample.get("katago_best", "N/A")

        print("\n" + "="*50)
        print("【当前局面 (User Prompt)】")
        print(user_prompt)
        print("="*50)

        # 组装 Prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_text += "<think>\n" # 强行引导进入思考

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        print("🤖 模型思考中...\n")
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1, # 极低温度，测试其确定性能力
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # 补齐我们强行加的 <think> 标签以便解析
        full_response = "<think>\n" + response 

        # 解析输出
        move = extract_move(full_response)
        think_text = extract_think_content(full_response)
        explain_text = extract_explanation(full_response)

        # 验证结果
        is_legal = move in katago_all
        model_wr = katago_all.get(move, 0.0) if is_legal else "N/A"

        print(f"🧠 【思考过程】:\n{think_text}\n")
        print(f"🗣️  【解说内容】:\n{explain_text}\n")
        print("-" * 50)
        print(f"♟️  【模型落子】: {move}")
        print(f"⚖️  【合法性】: {'✅ 是' if is_legal else '❌ 否 (不在 KataGo 候选点中)'}")
        print(f"📈 【胜率对比】: 模型落子胜率 = {model_wr} | 最佳胜率 = {katago_best_wr}")
        print("="*50)

if __name__ == "__main__":
    main()
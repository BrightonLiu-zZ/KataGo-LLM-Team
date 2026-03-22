import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 1. 核心路径配置 ---
BASE_MODEL = "Qwen/Qwen3-8B"
# LORA_PATH = "./runs/Qwen2.5-7B-GRPO-Go-Pro-v2/checkpoint-900"  # 请根据实际路径调整
# Switch between v3 and v4 checkpoints:
# LORA_PATH = "./runs/Qwen3-8B-GRPO-Go-Pro-v3/checkpoint-500"
LORA_PATH = "./runs/Qwen3-8B-GRPO-Go-Pro-v4/checkpoint-500"
print("🔋 正在以全精度 (BF16) 加载基座模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print(f"🧠 正在挂载 LoRA 权重: {LORA_PATH} ...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# --- 2. 准备系统提示词 ---
SYSTEM_PROMPT = (
    "You are a 9x9 Go (Weiqi) player. "
    "Board notation: '.' = empty, 'X' = Black, 'O' = White. "
    "Columns: A-J (no I). Rows: 1 (bottom) to 9 (top). "
    "You MUST only play on an empty '.' intersection listed in the valid coordinates.\n\n"
    "Respond in exactly this format:\n"
    "REASONING: [2-4 sentences analyzing the position]\n"
    "MOVE: [coordinate, e.g. D4]"
)



# --- 3. 交互式对话循环 ---
print("\n" + "="*50)
print("✅ 模型加载完毕！进入交互模式。")
print("👉 请在下方粘贴你的局势提示词（支持多行）。")
print("👉 粘贴完成后，在新的一行输入 'GO' 并回车，让模型开始思考。")
print("👉 输入 'QUIT' 退出程序。")
print("="*50 + "\n")

while True:
    print("\n📝 请粘贴棋盘 (输入 GO 提交, QUIT 退出):")
    
    user_prompt_lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
            
        if line.strip().upper() == 'QUIT':
            print("👋 退出测试。")
            exit()
        elif line.strip().upper() == 'GO':
            break
        else:
            user_prompt_lines.append(line)
            
    user_prompt = "\n".join(user_prompt_lines).strip()
    
    if not user_prompt:
        print("⚠️ 提示词不能为空，请重新输入。")
        continue

    # 组装消息
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    print("\n🚀 模型正在思考中...")
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print("\n" + "✨ 思考结果 ✨".center(40, "="))
    print(response)
    print("="*46)
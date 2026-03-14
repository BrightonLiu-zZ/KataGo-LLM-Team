import json
import random
import os

# ================= 配置区 =================
# 输入：上一以增强后的文件
INPUT_FILE = 'data/training_data_augmented_v3.jsonl'

# 输出：最终喂给模型的训练数据
OUTPUT_FILE = 'data/training_data_augmented_shuffled_v3.jsonl'

# 设置随机种子，保证每次打乱的结果一致（方便复现）
RANDOM_SEED = 42
# =========================================

def shuffle_dataset(input_path, output_path):
    print(f"正在读取文件: {input_path} ...")
    
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return

    data = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(line)
    except Exception as e:
        print(f"读取出错: {e}")
        return

    print(f"读取完成，共 {len(data)} 条数据。")
    print("正在进行全局打乱 (Global Shuffle)...")
    
    # 设置种子并打乱
    random.seed(RANDOM_SEED)
    random.shuffle(data)
    
    print(f"正在写入输出文件: {output_path} ...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in data:
                f.write(line + '\n')
    except Exception as e:
        print(f"写入出错: {e}")
        return

    print("="*30)
    print("打乱完成！Ready for Training.")
    print(f"输入: {len(data)} 行")
    print(f"输出: {len(data)} 行")
    print("="*30)
    
    # 验证一下打乱效果（打印前3个ID）
    print("验证前3条数据的ID分布:")
    for i in range(min(3, len(data))):
        try:
            record = json.loads(data[i])
            print(f"Line {i+1}: {record.get('original_id', 'Unknown')}")
        except:
            pass

if __name__ == "__main__":
    shuffle_dataset(INPUT_FILE, OUTPUT_FILE)
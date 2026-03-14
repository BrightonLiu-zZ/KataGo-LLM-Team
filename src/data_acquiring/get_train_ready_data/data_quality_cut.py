import json
import os

INPUT_FILE = 'data/training_ready_data_v3.jsonl'
OUTPUT_FILE = 'data/training_data_quality_filtered_v3.jsonl'

def curate_dataset():
    print("开始极简清洗：保留所有手数 >= 4 的对局...")
    count, kept = 0, 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip(): continue
            count += 1
            record = json.loads(line)
            
            # 新手科普：用 original_id 提取出了手数，例如 id_pos12 代表第12手
            move_num = -1
            if '_pos' in record.get("original_id", ""):
                try: move_num = int(record["original_id"].split('_pos')[-1])
                except: pass
            
            # 只要下了 4 手以上的对局，我们统统保留！
            if move_num >= 4:
                fout.write(json.dumps(record) + '\n')
                kept += 1
                
    print(f"清洗完成！共检查 {count} 条，保留了 {kept} 条高泛化数据。")

if __name__ == "__main__":
    curate_dataset()
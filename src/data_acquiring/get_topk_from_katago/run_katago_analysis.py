import subprocess
import json
import os
import sys
import time

# ================= 配置区 =================
# 1. 试运行限制：设为 10000 用于快速测试。
MAX_LIMIT = 10000

# 2. 核心路径 (请确认你的 9x9 模型文件名是否正确)
KATAGO_EXE = r"C:\git_repo\KataGo-LLM-Team\KataGo_engine\katago.exe"
CONFIG_FILE = r"C:\git_repo\KataGo-LLM-Team\script\get_topk_from_katago\analysis.cfg"
MODEL_FILE = r"C:\git_repo\KataGo-LLM-Team\KataGo_engine\KataGo18b9x9.gz"

# 3. 输入输出
INPUT_FILE = r"C:\git_repo\KataGo-LLM-Team\data\json_output.jsonl"
OUTPUT_FILE = r"C:\git_repo\KataGo-LLM-Team\data\json_output_with_topk.jsonl"

# 4. Top-K 设置: 我们在脚本里截取前几手
TOP_K = 10
# ===========================================

def run_analysis():
    cmd = [
        KATAGO_EXE, "analysis",
        "-config", CONFIG_FILE,
        "-model", MODEL_FILE
    ]
    
    print(f"=== KataGo 数据生成脚本 (Top-{TOP_K} 修正版) ===")
    print(f"目标数量: {MAX_LIMIT if MAX_LIMIT else '全部'}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            encoding='utf-8',
            bufsize=1 
        )
    except FileNotFoundError:
        print(f"错误: 找不到 {KATAGO_EXE}")
        return

    print("引擎已启动，开始清洗并处理数据...")
    start_time = time.time()
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        count = 0
        for line in fin:
            if not line.strip(): continue
            
            # 刹车机制
            if MAX_LIMIT and count >= MAX_LIMIT:
                print(f"\n[提示] 已达到设定限制 {MAX_LIMIT} 行，停止运行。")
                break

            try:
                original_data = json.loads(line)
                
                # === [关键修复] 数据清洗 ===
                # 不要直接发送 original_data，而是构造一个干净的 query
                # KataGo 只接受以下标准字段
                query = {
                    "id": original_data.get("id"),
                    "rules": original_data.get("rules", "chinese"),
                    "komi": original_data.get("komi", 6.5),
                    "boardXSize": original_data.get("boardXSize", 9),
                    "boardYSize": original_data.get("boardYSize", 9),
                    "initialStones": original_data.get("initialStones", []),
                    "moves": original_data.get("moves", []),
                    # 确保只分析当前这一手
                    "analyzeTurns": original_data.get("analyzeTurns", [len(original_data.get("moves", []))])
                }
                
                # 发送干净的 Query
                process.stdin.write(json.dumps(query) + "\n")
                process.stdin.flush()
                
                # 获取结果
                response_line = process.stdout.readline()
                if not response_line:
                    print("\n引擎异常退出。")
                    break
                    
                response = json.loads(response_line)
                
                if "error" in response:
                    print(f"\n警告: ID {query['id']} 出错: {response['error']}")
                elif "warning" in response:
                    # 如果还有警告，打印出来看看
                    print(f"\n警告: ID {query['id']} : {response['warning']}")

                # === [关键步骤] 提取 Top-K ===
                # KataGo 返回的 moveInfos 包含了所有候选点的信息
                if "moveInfos" in response:
                    # 截取前 10 手
                    top_moves = response["moveInfos"][:TOP_K]
                    
                    # 将提取好的 Top-K 数据塞回原始数据中
                    # 这样 LLM 就能直接读到 "candidate_moves" 列表
                    original_data["katago_analysis"] = top_moves
                else:
                    original_data["katago_analysis"] = response # 如果出错，保留原始错误信息

                # 写入文件
                fout.write(json.dumps(original_data, ensure_ascii=False) + "\n")
                
                count += 1
                if count % 10 == 0:
                    elapsed = time.time() - start_time
                    speed = count / elapsed if elapsed > 0 else 0
                    print(f"进度: {count} | 速度: {speed:.1f} 图/秒", end='\r')
                    
            except Exception as e:
                print(f"\n处理出错: {e}")
                break

    process.stdin.close()
    process.terminate()
    print(f"\n\n=== 完成 ===")
    print(f"输出文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_analysis()
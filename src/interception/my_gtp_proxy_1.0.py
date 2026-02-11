import sys
import subprocess
import threading
import queue
import json
import time
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 准备记录全局数据
game_data = {
    "boardsize": None,
    "komi": None,
    "moves": [],     # 存储 [color, move] 的列表，比如 [B, Q16]
    "analysis": {}   # 字典，key为move
}

# 保存计数器
save_counter = 0
SAVE_INTERVAL = 5  # 每 5 次更新保存一次

# 用于跟踪已下的棋子位置
played_moves = set()

def save_game_data():
    """
    将 game_data 保存到 game_data.json 文件，覆盖写入，确保文件始终包含最新状态。
    修改点：按胜率（winrate）降序排序，保留前 5 条分析数据。
    """
    global save_counter
    save_counter += 1
    if save_counter % SAVE_INTERVAL != 0:  # 每 SAVE_INTERVAL 次保存一次
        return

    # 转换为列表，按 winrate 排序，保留前 5 条
    analysis_list = list(game_data["analysis"].values())
    analysis_list.sort(key=lambda x: x.get("winrate", 0), reverse=True)
    top_analysis = analysis_list[:5]  # 取前 5 条，按 winrate 排序. noted: 这里不知道因为什么bug没有实现这个功能

    game_data_to_save = {
        "boardsize": game_data["boardsize"],
        "komi": game_data["komi"],
        "moves": game_data["moves"],
        "analysis": top_analysis
    }

    try:
        with open("game_data.json", "w", encoding='utf-8') as f:
            json.dump(game_data_to_save, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved game_data to game_data.json: {json.dumps(game_data_to_save, ensure_ascii=False)}")
    except Exception as e:
        logger.error(f"Error saving game_data: {e}")

def parse_gtp_command(command_line):
    """
    解析Lizzie发来的GTP命令（比如play B Q16），更新棋局信息（game_data和played_moves）。
    """
    logger.info(f"[DEBUG] Received command: {command_line}")
    tokens = command_line.split()
    if not tokens:
        return

    cmd = tokens[0].lower()

    if cmd == "play" and len(tokens) >= 3:
        color = tokens[1]
        move = tokens[2]
        game_data["moves"].append([color, move])
        game_data["analysis"] = {}  # 清空分析，等待主循环发送“kata-analyze”
        if move != "pass":  # 如果不是“pass”，记录到 played_moves
            played_moves.add(move)  # 添加新下的棋子位置
    elif cmd == "boardsize" and len(tokens) >= 2:
        game_data["boardsize"] = int(tokens[1])
    elif cmd == "komi" and len(tokens) >= 2:
        game_data["komi"] = float(tokens[1])
    elif cmd == "clear_board":
        game_data["moves"].clear()
        game_data["analysis"].clear()
        played_moves.clear()  # 清空已下棋的着点
    elif cmd == "undo":
        if game_data["moves"]:
            last_move = game_data["moves"].pop()  # 移除最后一手棋
            if last_move[1] != "pass":  # 如果不是“pass”，从 played_moves 移除
                played_moves.remove(last_move[1])  # 移除最后下的棋子位置
            game_data["analysis"] = {}  # 清空分析

    # 每次解析后记录 game_data 到 JSON 文件（依赖计数器）
    logger.info(f"[DEBUG] Updated game_data: {json.dumps(game_data, ensure_ascii=False)}")
    save_game_data()

def parse_katago_info_line(line):
    """
    解析KataGo的分析输出（比如info move Q16 visits 123 winrate 0.55 pv Q16 D4），
    只保留 move、winrate 和 pv 。
    """
    parts = line.strip().split()
    if len(parts) < 3 or parts[0] != "info" or parts[1] != "move":
        return None

    idx = 2
    move = parts[idx]
    idx += 1
    analysis_record = {"move": move}

    while idx < len(parts):
        key = parts[idx]
        if key == "pv":
            break
        idx += 1
        if idx >= len(parts):
            break
        value = parts[idx]
        if key == "winrate":  # 只保留 winrate
            try:
                analysis_record[key] = float(value)
            except ValueError:
                analysis_record[key] = value
        idx += 1

    if idx < len(parts) and parts[idx] == "pv":
        idx += 1
        pv_moves = []
        while idx < len(parts):
            pv_moves.append(parts[idx])
            idx += 1
        if pv_moves:
            analysis_record["pv"] = pv_moves[:10]  # 限制 pv 为前 10 步

    return analysis_record

def main():
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("Usage: python my_gtp_proxy.py [KataGo_executable_and_args]")
        sys.exit(1)

    katago_command = sys.argv[1:]
    logger.info(f"[PROXY]: Starting KataGo: {' '.join(katago_command)}")

    # 启动 KataGo 进程
    katago_proc = subprocess.Popen(
        katago_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True
    )

    # 使用Queue存储 KataGo 输出
    katago_stdout_queue = queue.Queue()
    katago_stderr_queue = queue.Queue()

    def read_katago_output():
        """持续读取 KataGo 的 stdout 输出"""
        while True:
            line = katago_proc.stdout.readline()
            if not line:
                break
            logger.info(f"[DEBUG] KataGo stdout: {line.strip()}")
            if line.startswith("info move"):
                record = parse_katago_info_line(line)
                if record:
                    move = record["move"]
                    game_data["analysis"][move] = record  # 更新或添加记录
                    logger.info(f"[DEBUG] Parsed analysis: {record}")
                    save_game_data()  # 每次分析后保存到 game_data.json
            katago_stdout_queue.put(line)

    def read_katago_stderr():
        """线程函数：持续读取 KataGo 的 stderr 输出"""
        while True:
            line = katago_proc.stderr.readline()
            if not line:
                break
            logger.info(f"[DEBUG] KataGo stderr: {line.strip()}")
            katago_stderr_queue.put(line)

    # 启动读取线程
    stdout_thread = threading.Thread(target=read_katago_output, daemon=True)
    stderr_thread = threading.Thread(target=read_katago_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    try:
        # 主循环：处理 Lizzie 的 GTP 命令
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.rstrip("\r\n")
            if not line:
                continue

            # 解析并记录命令
            parse_gtp_command(line)

            # 转发给 KataGo 并处理响应
            katago_proc.stdin.write(line + "\n")
            katago_proc.stdin.flush()

            tokens = line.split()
            if tokens and tokens[0].lower() == "play":
                # 等待 KataGo 响应“play”命令
                start_time = time.time()
                while True:
                    try:
                        out_line = katago_stdout_queue.get(timeout=0.1)
                        sys.stdout.write(out_line)
                        sys.stdout.flush()
                    except queue.Empty:
                        if time.time() - start_time > 1.0:  # 等待 1 秒
                            break

                # 确定下一手棋的颜色并发送“kata-analyze”
                last_color = game_data["moves"][-1][0] if game_data["moves"] else None
                next_color = "W" if last_color == "B" else "B"
                katago_proc.stdin.write(f"kata-analyze {next_color} 100\n")
                katago_proc.stdin.flush()
                logger.info(f"Sent kata-analyze {next_color} 100 to KataGo")

            if line.lower().startswith("quit"):
                # 保存 game_data 到 JSON 文件
                save_game_data()
                # 关闭 KataGo 的输入管道
                katago_proc.stdin.close()
                # 读取所有剩余输出
                while True:
                    try:
                        out_line = katago_stdout_queue.get(block=False)
                        sys.stdout.write(out_line)
                        sys.stdout.flush()
                    except queue.Empty:
                        break
                # 读取所有剩余错误输出
                while True:
                    try:
                        err_line = katago_stderr_queue.get(block=False)
                        sys.stderr.write(err_line)
                        sys.stderr.flush()
                    except queue.Empty:
                        break
                # 等待 KataGo 进程终止
                katago_proc.wait(timeout=5)
                # 确保所有管道关闭
                katago_proc.stdout.close()
                katago_proc.stderr.close()
                break
            else:
                # 其他命令的处理逻辑：持续读取并转发所有输出
                start_time = time.time()
                while True:
                    try:
                        out_line = katago_stdout_queue.get(timeout=0.1)
                        sys.stdout.write(out_line)
                        sys.stdout.flush()
                    except queue.Empty:
                        if time.time() - start_time > 2.0:  # 2 秒的等待时间
                            break

    except Exception as e:
        logger.error(f"Exception occurred: {e}")

    finally:
        # 确保进程被终止
        try:
            katago_proc.terminate()
            katago_proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            katago_proc.kill()
        except:
            pass
        # 确保所有管道关闭
        try:
            katago_proc.stdin.close()
        except:
            pass
        try:
            katago_proc.stdout.close()
        except:
            pass
        try:
            katago_proc.stderr.close()
        except:
            pass
        # 等待线程结束
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

if __name__ == "__main__":
    main()

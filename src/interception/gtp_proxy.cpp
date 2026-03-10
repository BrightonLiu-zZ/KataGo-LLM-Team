#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <mutex>

// 引入我们的两大神器
#include "include/nlohmann/json.hpp"
// #define CPPHTTPLIB_OPENSSL_SUPPORT // 如果后续需要 HTTPS 可以开启（需配置 OpenSSL）
#include "include/httplib.h"

using namespace std;
using json = nlohmann::json;

// --- 游戏状态 ---
struct GameState {
    int boardsize = 9; // 我们的模型是 9x9 的
    vector<pair<string, string>> moves; // 记录类似 ("B", "Q16")
};

static GameState g_state;
static mutex g_stateMutex;

// --- 工具函数 ---
static string trim(const string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// 提取大模型回答中的坐标
static string extract_answer(const string& text) {
    size_t start = text.find("<answer>");
    size_t end = text.find("</answer>");
    if (start != string::npos && end != string::npos) {
        start += 8; // 跳过 "<answer>"
        return trim(text.substr(start, end - start));
    }
    return "pass"; // 如果模型发疯没按格式输出，默认 pass 防崩溃
}

// --- 核心：请求本地 LLM ---
static string ask_local_llm(const string& color_to_play) {
    // 1. 组装 System Prompt (严格使用你训练时的设定)
    string system_prompt = 
        "You are an expert 9x9 Go (Weiqi) Player. You must determine the best tactical next move.\n"
        "The board is a 9x9 grid. Columns are A through J (skipping I), and rows are 1 through 9.\n"
        "In the provided board state, '.' represents empty, 'X' is Black, and 'O' is White.\n"
        "Your task:\n"
        "1. Think about the current board, capture threats, territories, and life-and-death.\n"
        "2. Write your analysis inside <think></think> tags.\n"
        "3. Finally, output YOUR CHOSEN COORDINATE (e.g., C4) strictly inside <answer></answer> tags.";

    // 2. 组装 User Prompt (由于无法轻易画图，采用极简历史记录拼接)
    string user_prompt = "[Current 9x9 Board State]\n(Board visualization omitted for edge demo)\nLast moves: ";
    for (size_t i = 0; i < g_state.moves.size(); ++i) {
        user_prompt += g_state.moves[i].first + " " + g_state.moves[i].second;
        if (i != g_state.moves.size() - 1) user_prompt += "; ";
    }
    string full_color = (color_to_play == "B" || color_to_play == "b" || color_to_play == "black") ? "Black" : "White";
    user_prompt += "\n\nPlayer to move: " + full_color + ".";

    // 3. 构建发给 LM Studio 的 JSON Payload
    json payload = {
        {"model", "qwen-7b-go-Q4_K_M.gguf"}, // 确保和 LM Studio 里加载的模型名字对应
        {"messages", json::array({
            {{"role", "system"}, {"content", system_prompt}},
            {{"role", "user"}, {"content", user_prompt}}
        })},
        {"temperature", 0.1}, // 降低温度，保证下棋逻辑严密
        {"max_tokens", 1024}
    };

    // 4. 发起 HTTP POST 请求到本地 1234 端口
    httplib::Client cli("localhost", 1234);
    cli.set_read_timeout(120); // 给模型最多 2 分钟的思考时间
    
    auto res = cli.Post("/v1/chat/completions", payload.dump(), "application/json");
    
    // 5. 解析结果
    if (res && res->status == 200) {
        json response_json = json::parse(res->body);
        string model_reply = response_json["choices"][0]["message"]["content"];
        return extract_answer(model_reply);
    } else {
        return "pass"; // 网络错误时默认 pass
    }
}

// --- 主循环 ---
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    string line;
    while (getline(cin, line)) {
        line = trim(line);
        if (line.empty()) continue;

        stringstream ss(line);
        string cmd;
        ss >> cmd;

        if (cmd == "name") { cout << "= Qwen-Edge-Engine\n\n"; } 
        else if (cmd == "version") { cout << "= 1.0\n\n"; } 
        else if (cmd == "protocol_version") { cout << "= 2\n\n"; } 
        else if (cmd == "list_commands") {
            cout << "= name\nversion\nprotocol_version\nboardsize\nclear_board\nplay\ngenmove\nquit\n\n";
        } 
        else if (cmd == "boardsize") {
            int size = 9;
            ss >> size;
            lock_guard<mutex> lock(g_stateMutex);
            g_state.boardsize = size;
            cout << "=\n\n"; 
        } 
        else if (cmd == "clear_board") {
            lock_guard<mutex> lock(g_stateMutex);
            g_state.moves.clear();
            cout << "=\n\n";
        } 
        else if (cmd == "play") {
            string color, coord;
            ss >> color >> coord;
            lock_guard<mutex> lock(g_stateMutex);
            g_state.moves.push_back({color, coord});
            cout << "=\n\n";
        } 
        else if (cmd == "genmove") {
            string color;
            ss >> color;
            // 阻断式调用：去本地 LM Studio 问结果
            string best_move = ask_local_llm(color);
            
            // 将自己思考的落子也加入历史记录
            {
                lock_guard<mutex> lock(g_stateMutex);
                g_state.moves.push_back({color, best_move});
            }
            cout << "= " << best_move << "\n\n";
        } 
        else if (cmd == "quit") {
            cout << "=\n\n";
            break;
        } 
        else {
            cout << "? unknown command\n\n";
        }

        cout.flush(); 
    }
    return 0;
}
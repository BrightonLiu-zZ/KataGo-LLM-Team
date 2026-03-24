#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <array>
#include <algorithm>
#include <regex>
#include <cctype>
#include <cstring>

#include "include/nlohmann/json.hpp"
#include "include/httplib.h"

using namespace std;
using json = nlohmann::json;

// ===================== Board Constants =====================

static const int BOARD_SIZE = 9;
static const char VALID_COLS[] = "ABCDEFGHJ"; // skip I

enum Cell : char { EMPTY = '.', BLACK = 'X', WHITE = 'O' };

// ===================== Board Engine =====================

struct Board {
    array<array<char, BOARD_SIZE>, BOARD_SIZE> grid;  // grid[row][col], row 0 = row 1 (bottom)
    vector<pair<string, string>> move_history;        // ("B","D4"), ("W","E5"), ...

    Board() { clear(); }

    void clear() {
        for (auto& row : grid)
            row.fill(EMPTY);
        move_history.clear();
    }

    // Convert GTP coordinate (e.g. "D4") to internal (col, row) indices.
    // Returns false if the coordinate is invalid or "pass".
    bool parse_coord(const string& coord, int& col, int& row) const {
        if (coord.size() < 2) return false;
        char c = toupper(coord[0]);
        const char* p = strchr(VALID_COLS, c);
        if (!p) return false;
        col = static_cast<int>(p - VALID_COLS);
        row = atoi(coord.c_str() + 1) - 1;  // "1" -> index 0
        if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE)
            return false;
        return true;
    }

    static char color_to_cell(const string& color) {
        char c = toupper(color[0]);
        return (c == 'B') ? BLACK : WHITE;
    }

    static char opponent(char cell) {
        return (cell == BLACK) ? WHITE : BLACK;
    }

    // Flood-fill to find a connected group and its liberty count.
    void flood(int r, int c, char stone,
               vector<pair<int,int>>& group,
               int& liberties,
               array<array<bool, BOARD_SIZE>, BOARD_SIZE>& visited) const {
        if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) return;
        if (visited[r][c]) return;
        if (grid[r][c] == EMPTY) { liberties++; visited[r][c] = true; return; }
        if (grid[r][c] != stone) return;
        visited[r][c] = true;
        group.push_back({r, c});
        flood(r-1, c, stone, group, liberties, visited);
        flood(r+1, c, stone, group, liberties, visited);
        flood(r, c-1, stone, group, liberties, visited);
        flood(r, c+1, stone, group, liberties, visited);
    }

    // Remove a group (set cells to EMPTY).
    void remove_group(const vector<pair<int,int>>& group) {
        for (auto [r, c] : group)
            grid[r][c] = EMPTY;
    }

    // Place a stone and handle captures. Returns true on success.
    bool play(const string& color, const string& coord) {
        string upper_coord = coord;
        for (auto& ch : upper_coord) ch = toupper(ch);

        string short_color;
        char c0 = toupper(color[0]);
        short_color = (c0 == 'B') ? "B" : "W";

        if (upper_coord == "PASS") {
            move_history.push_back({short_color, "pass"});
            return true;
        }

        int col, row;
        if (!parse_coord(upper_coord, col, row)) return false;
        if (grid[row][col] != EMPTY) return false;

        char stone = color_to_cell(color);
        char opp   = opponent(stone);
        grid[row][col] = stone;

        // Check and remove captured opponent groups adjacent to the placed stone.
        static const int dr[] = {-1, 1, 0, 0};
        static const int dc[] = {0, 0, -1, 1};
        for (int d = 0; d < 4; d++) {
            int nr = row + dr[d], nc = col + dc[d];
            if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) continue;
            if (grid[nr][nc] != opp) continue;
            array<array<bool, BOARD_SIZE>, BOARD_SIZE> visited{};
            vector<pair<int,int>> group;
            int liberties = 0;
            flood(nr, nc, opp, group, liberties, visited);
            if (liberties == 0)
                remove_group(group);
        }

        // Self-capture check (suicide): remove own group if zero liberties.
        {
            array<array<bool, BOARD_SIZE>, BOARD_SIZE> visited{};
            vector<pair<int,int>> group;
            int liberties = 0;
            flood(row, col, stone, group, liberties, visited);
            if (liberties == 0)
                remove_group(group);
        }

        move_history.push_back({short_color, upper_coord});
        return true;
    }

    // ---- Rendering ----

    string render_ascii() const {
        ostringstream os;
        os << "[Current 9x9 Board State]\n";
        os << "   A B C D E F G H J\n";
        for (int r = BOARD_SIZE - 1; r >= 0; r--) {
            os << " " << (r + 1);
            for (int c = 0; c < BOARD_SIZE; c++)
                os << " " << grid[r][c];
            os << "\n";
        }
        return os.str();
    }

    string render_move_history() const {
        if (move_history.empty()) return "";
        ostringstream os;
        for (size_t i = 0; i < move_history.size(); i++) {
            if (i > 0) os << "; ";
            os << move_history[i].first << " " << move_history[i].second;
        }
        return os.str();
    }

    // Valid empty coordinates sorted by (row asc, col asc) — matches prepare_v4_dataset.py
    string render_valid_coords() const {
        ostringstream os;
        bool first = true;
        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                if (grid[r][c] == EMPTY) {
                    if (!first) os << ", ";
                    os << VALID_COLS[c] << (r + 1);
                    first = false;
                }
            }
        }
        return os.str();
    }

    // Build the full user prompt matching v4 training format.
    string build_user_prompt(const string& color_to_play) const {
        string full_color = (color_to_play == "B" || color_to_play == "b" ||
                             color_to_play == "black" || color_to_play == "BLACK")
                            ? "Black" : "White";

        ostringstream os;
        os << render_ascii();
        os << "\nLast moves: " << render_move_history() << "\n";
        os << "\nPlayer to move: " << full_color << ". \n";
        os << "Analyze the board, then pick a move and explain your reasoning.\n";
        os << "\nValid empty coordinates: [" << render_valid_coords() << "]\n";
        return os.str();
    }
};

// ===================== LLM Interface =====================

static const char* SYSTEM_PROMPT =
    "You are a 9x9 Go (Weiqi) player. "
    "Board notation: '.' = empty, 'X' = Black, 'O' = White. "
    "Columns: A-J (no I). Rows: 1 (bottom) to 9 (top). "
    "You MUST only play on an empty '.' intersection listed in the valid coordinates.\n\n"
    "Respond in exactly this format:\n"
    "REASONING: [2-4 sentences analyzing the position]\n"
    "MOVE: [coordinate, e.g. D4]\n\n"
    "/no_think";

static const char* LM_STUDIO_MODEL = "qwen/qwen3-8b@q4_k_m";
static const char* LM_STUDIO_HOST  = "localhost";
static const int   LM_STUDIO_PORT  = 1234;

static string trim(const string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Remove <think>...</think> blocks, returning only the visible model output.
static string strip_think_tags(const string& text) {
    string result = text;
    while (true) {
        size_t start = result.find("<think>");
        if (start == string::npos) break;
        size_t end = result.find("</think>", start);
        if (end == string::npos) break;
        result.erase(start, end + 8 - start); // 8 = strlen("</think>")
    }
    return trim(result);
}

// Extract coordinate from "MOVE: D4" (matches train_grpo_v4.py extract_move)
static string extract_move(const string& text) {
    regex re(R"(MOVE:\s*([A-HJa-hj][1-9]|PASS|pass))", regex::icase);
    smatch m;
    if (regex_search(text, m, re)) {
        string mv = m[1].str();
        for (auto& ch : mv) ch = toupper(ch);
        return mv;
    }
    return "pass";
}

static string ask_local_llm(Board& board, const string& color_to_play) {
    string user_prompt = board.build_user_prompt(color_to_play);

    json payload = {
        {"model", LM_STUDIO_MODEL},
        {"messages", json::array({
            {{"role", "system"}, {"content", SYSTEM_PROMPT}},
            {{"role", "user"},   {"content", user_prompt}}
        })},
        {"temperature", 0.1},
        {"max_tokens", 4096},
        {"chat_template_kwargs", {{"enable_thinking", false}}}
    };

    httplib::Client cli(LM_STUDIO_HOST, LM_STUDIO_PORT);
    cli.set_read_timeout(120);

    auto res = cli.Post("/v1/chat/completions", payload.dump(), "application/json");

    if (res && res->status == 200) {
        json resp = json::parse(res->body);
        return resp["choices"][0]["message"]["content"].get<string>();
    }
    return "";
}

// ===================== GTP Main Loop =====================

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    Board board;
    string line;

    while (getline(cin, line)) {
        line = trim(line);
        if (line.empty()) continue;

        stringstream ss(line);
        string cmd;
        ss >> cmd;
        for (auto& ch : cmd) ch = tolower(ch);

        if (cmd == "name") {
            cout << "= Qwen3-Go-v4\n\n";
        }
        else if (cmd == "version") {
            cout << "= 2.0\n\n";
        }
        else if (cmd == "protocol_version") {
            cout << "= 2\n\n";
        }
        else if (cmd == "list_commands") {
            cout << "= name\nversion\nprotocol_version\nboardsize\n"
                    "clear_board\nplay\ngenmove\nkomi\nknown_command\nquit\n\n";
        }
        else if (cmd == "known_command") {
            string qcmd;
            ss >> qcmd;
            for (auto& ch : qcmd) ch = tolower(ch);
            bool known = (qcmd == "name" || qcmd == "version" ||
                          qcmd == "protocol_version" || qcmd == "list_commands" ||
                          qcmd == "boardsize" || qcmd == "clear_board" ||
                          qcmd == "play" || qcmd == "genmove" ||
                          qcmd == "komi" || qcmd == "known_command" ||
                          qcmd == "quit");
            cout << "= " << (known ? "true" : "false") << "\n\n";
        }
        else if (cmd == "boardsize") {
            int size = 9;
            ss >> size;
            if (size != 9) {
                cout << "? unacceptable size\n\n";
            } else {
                board.clear();
                cout << "=\n\n";
            }
        }
        else if (cmd == "clear_board") {
            board.clear();
            cout << "=\n\n";
        }
        else if (cmd == "komi") {
            // Accept and ignore — the model doesn't use komi.
            cout << "=\n\n";
        }
        else if (cmd == "play") {
            string color, coord;
            ss >> color >> coord;
            if (board.play(color, coord)) {
                cout << "=\n\n";
            } else {
                cout << "? illegal move\n\n";
            }
        }
        else if (cmd == "genmove") {
            string color;
            ss >> color;
            string full_reply = ask_local_llm(board, color);
            string best = extract_move(full_reply);
            string reasoning = strip_think_tags(full_reply);
            if (!reasoning.empty())
                cerr << "[Qwen3] " << reasoning << endl;
            if (!board.play(color, best)) {
                best = "pass";
                board.play(color, best);
            }
            cout << "= " << best << "\n\n";
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

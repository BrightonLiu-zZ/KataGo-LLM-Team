#include <windows.h>
#include <io.h>
#include <fcntl.h>

#include <cctype>   // added
#include <cstdio>   // added
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// Game state
// ---------------------------------------------------------------------------
struct GameState {
    int         boardsize = 19;
    double      komi      = 7.5;
    // Each entry: (color, coordinate), e.g. ("B", "Q16")
    std::vector<std::pair<std::string, std::string>> moves;
};

static GameState  g_state;
static std::mutex g_stateMutex;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string to_upper(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return {};
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------
static void save_game_data() {
    // Called with g_stateMutex held.
    nlohmann::json j;
    j["board_size"] = std::to_string(g_state.boardsize);
    j["komi"]       = g_state.komi;

    auto& arr = j["moves"];
    arr = nlohmann::json::array();
    for (size_t i = 0; i < g_state.moves.size(); ++i) {
        const auto& [color, coord] = g_state.moves[i];
        std::string color_name = (to_upper(color) == "B") ? "Black" : "White";
        std::string desc = "Move " + std::to_string(i + 1) + ": "
                         + color_name + " plays " + coord;
        arr.push_back(desc);
    }

    std::ofstream ofs("game_data.json", std::ios::binary);
    if (ofs) {
        ofs << j.dump(2) << '\n';
    }
}

// ---------------------------------------------------------------------------
// GTP command parser – updates game state, returns true if state changed
// ---------------------------------------------------------------------------
static bool parse_gtp_command(const std::string& line) {
    std::istringstream iss(line);
    std::string cmd;
    if (!(iss >> cmd)) return false;

    cmd = to_upper(cmd);

    // Skip optional numeric id prefix (e.g. "1 play B Q16")
    if (!cmd.empty() && std::isdigit(static_cast<unsigned char>(cmd[0]))) {
        if (!(iss >> cmd)) return false;
        cmd = to_upper(cmd);
    }

    std::lock_guard<std::mutex> lock(g_stateMutex);

    if (cmd == "BOARDSIZE") {
        int sz = 0;
        if (iss >> sz) g_state.boardsize = sz;
        return true;
    }
    if (cmd == "KOMI") {
        double k = 0.0;
        if (iss >> k) g_state.komi = k;
        return true;
    }
    if (cmd == "PLAY") {
        std::string color, coord;
        if (iss >> color >> coord) {
            g_state.moves.emplace_back(to_upper(color), to_upper(coord));
            return true;
        }
        return false;
    }
    if (cmd == "UNDO") {
        if (!g_state.moves.empty()) {
            g_state.moves.pop_back();
        }
        return true;
    }
    if (cmd == "CLEAR_BOARD") {
        g_state.moves.clear();
        return true;
    }

    return false;  // unrecognized / irrelevant command
}

// ---------------------------------------------------------------------------
// Pipe helpers
// ---------------------------------------------------------------------------

// Write a line (with newline) to a pipe HANDLE.
static bool write_pipe(HANDLE h, const std::string& line) {
    std::string data = line + "\n";
    DWORD written = 0;
    return WriteFile(h, data.data(), static_cast<DWORD>(data.size()), &written, nullptr) != 0;
}

// Read from pipe HANDLE, write to a FILE* (stdout or stderr). Runs until pipe
// closes. This is the relay thread body.
static void relay_thread(HANDLE hRead, FILE* dest) {
    constexpr DWORD BUF_SZ = 4096;
    char buf[BUF_SZ];
    DWORD bytesRead = 0;
    while (ReadFile(hRead, buf, BUF_SZ, &bytesRead, nullptr) && bytesRead > 0) {
        fwrite(buf, 1, bytesRead, dest);
        fflush(dest);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: gtp_proxy.exe <katago_exe> [args...]\n"
                  << "Example: gtp_proxy.exe katago.exe gtp -model net.gz -config cfg.cfg\n";
        return 1;
    }

    // ---- Build child command line ----------------------------------------
    std::string cmdline;
    for (int i = 1; i < argc; ++i) {
        if (i > 1) cmdline += ' ';
        // Quote arguments that contain spaces
        std::string arg = argv[i];
        if (arg.find(' ') != std::string::npos) {
            cmdline += '"' + arg + '"';
        } else {
            cmdline += arg;
        }
    }

    // ---- Create pipes ----------------------------------------------------
    SECURITY_ATTRIBUTES sa{};
    sa.nLength        = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE hChildStdinRead   = nullptr;
    HANDLE hChildStdinWrite  = nullptr;
    HANDLE hChildStdoutRead  = nullptr;
    HANDLE hChildStdoutWrite = nullptr;
    HANDLE hChildStderrRead  = nullptr;
    HANDLE hChildStderrWrite = nullptr;

    if (!CreatePipe(&hChildStdinRead,   &hChildStdinWrite,  &sa, 0) ||
        !CreatePipe(&hChildStdoutRead,  &hChildStdoutWrite, &sa, 0) ||
        !CreatePipe(&hChildStderrRead,  &hChildStderrWrite, &sa, 0)) {
        std::cerr << "Failed to create pipes.\n";
        return 1;
    }

    // Prevent child from inheriting our ends of the pipes.
    SetHandleInformation(hChildStdinWrite,  HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(hChildStdoutRead,  HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(hChildStderrRead,  HANDLE_FLAG_INHERIT, 0);

    // ---- Launch KataGo ----------------------------------------------------
    STARTUPINFOA si{};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdInput  = hChildStdinRead;
    si.hStdOutput = hChildStdoutWrite;
    si.hStdError  = hChildStderrWrite;

    PROCESS_INFORMATION pi{};

    // CreateProcessA needs a mutable char buffer for lpCommandLine.
    std::vector<char> cmdBuf(cmdline.begin(), cmdline.end());
    cmdBuf.push_back('\0');

    if (!CreateProcessA(nullptr, cmdBuf.data(), nullptr, nullptr,
                        TRUE, 0, nullptr, nullptr, &si, &pi)) {
        std::cerr << "Failed to start KataGo: " << GetLastError() << '\n';
        return 1;
    }

    // Close child-side handles in parent.
    CloseHandle(hChildStdinRead);
    CloseHandle(hChildStdoutWrite);
    CloseHandle(hChildStderrWrite);

    // ---- Relay threads ---------------------------------------------------
    std::thread convergOut(relay_thread, hChildStdoutRead, stdout);
    std::thread convergErr(relay_thread, hChildStderrRead, stderr);

    // ---- Main loop: read from Lizzie (stdin) -----------------------------
    // Use binary mode so we control newlines ourselves.
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) continue;

        // Parse and possibly update game state + save JSON.
        if (parse_gtp_command(line)) {
            std::lock_guard<std::mutex> lock(g_stateMutex);
            save_game_data();
        }

        // Forward the command to KataGo.
        if (!write_pipe(hChildStdinWrite, line)) break;

        // On quit, stop sending further commands.
        if (to_upper(line.substr(0, 4)) == "QUIT") break;
    }

    // ---- Shutdown --------------------------------------------------------
    CloseHandle(hChildStdinWrite);          // signals EOF to KataGo
    WaitForSingleObject(pi.hProcess, 5000); // wait up to 5 s

    convergOut.join();
    convergErr.join();

    CloseHandle(hChildStdoutRead);
    CloseHandle(hChildStderrRead);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return 0;
}

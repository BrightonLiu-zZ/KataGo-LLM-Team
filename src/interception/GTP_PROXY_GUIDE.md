# GTP Proxy — User Guide

A beginner-friendly, step-by-step guide to using `gtp_proxy.exe`.

---

## What is this?

`gtp_proxy.exe` sits **between** Lizzie (a Go GUI) and KataGo (the AI engine).  
It silently records every move that Lizzie sends to KataGo and saves the game record to a file called `game_data.json`, **in real time, after every move**.

```
┌────────┐   GTP commands    ┌───────────────┐   forwarded    ┌────────┐
│ Lizzie │ ───────────────►  │ gtp_proxy.exe │ ────────────►  │ KataGo │
│  (GUI) │ ◄───────────────  │  (this tool)  │ ◄────────────  │(engine)│
└────────┘   GTP responses   └──────┬────────┘   responses    └────────┘
                                    │
                                    ▼
                            game_data.json
```

- Lizzie does **not** know the proxy exists — everything works exactly as before.  
- KataGo does **not** know the proxy exists — it receives the same commands.  
- The proxy is fully transparent; it just writes `game_data.json` as a side effect.

---

## Files you need

All files should be in the **same folder** (e.g. `C:\katago_old\lizzie\`):

| File | What it is |
|------|-----------|
| `gtp_proxy.exe` | The compiled proxy program (already built) |
| `my_gtp_proxy_cpp.bat` | A small script that launches the proxy with the right arguments |
| `katago.exe` | The KataGo engine (you should already have this) |
| `KataGo15b.gz` | The neural network weights file (you should already have this) |
| `default_gtp.cfg` | KataGo's config file (you should already have this) |

---

## Setup — Step by step

### Step 1: Verify all files are present

Open the folder `C:\katago_old\lizzie\` in File Explorer and confirm you can see:

- `gtp_proxy.exe`
- `my_gtp_proxy_cpp.bat`
- `katago.exe`
- `KataGo15b.gz`
- `default_gtp.cfg`

If `gtp_proxy.exe` is missing, see the **"How to compile"** section at the bottom.

### Step 2: Edit the bat file (if your paths are different)

Open `my_gtp_proxy_cpp.bat` in a text editor (right-click → Edit, or open with Notepad). It looks like this:

```bat
@echo off
cd /d %~dp0
gtp_proxy.exe C:\katago_old\lizzie\katago.exe gtp -model C:\katago_old\lizzie\KataGo15b.gz -config C:\katago_old\lizzie\default_gtp.cfg
```

**If your files are in `C:\katago_old\lizzie\`**, you don't need to change anything.

**If your files are somewhere else**, replace the three paths:
1. `C:\katago_old\lizzie\katago.exe` → your actual path to `katago.exe`
2. `C:\katago_old\lizzie\KataGo15b.gz` → your actual path to the weights file
3. `C:\katago_old\lizzie\default_gtp.cfg` → your actual path to the config file

Save and close.

### Step 3: Configure Lizzie to use the proxy

1. Open Lizzie.
2. Go to **Settings** (or the config file, depending on your Lizzie version).
3. Find the **Engine Command** field (the command line that tells Lizzie how to start KataGo).
4. Replace the current command with:

```
"C:\katago_old\lizzie\my_gtp_proxy_cpp.bat"
```

> **Important:** Include the double quotes around the path.

> **Note:** Lizzie may pass extra arguments like `gtp -model ... -config ...` after the command. That's fine — the bat file already contains the correct arguments, so the extra ones from Lizzie are harmless.

5. Save and restart Lizzie.

### Step 4: Play a game and check the output

1. Start Lizzie normally and play some moves (or let KataGo analyze a position).
2. Open the folder `C:\katago_old\lizzie\` in File Explorer.
3. You should see a file called `game_data.json`.
4. Open it with any text editor. It will look like this:

```json
{
  "board_size": "19",
  "komi": 7.5,
  "moves": [
    "Move 1: Black plays Q16",
    "Move 2: White plays D4",
    "Move 3: Black plays R14"
  ]
}
```

This file updates **automatically after every move** — you can keep it open and refresh to see changes.

---

## What gets recorded

| GTP Command | What happens |
|-------------|-------------|
| `boardsize 19` | Sets the board size in the JSON (e.g. `"19"`) |
| `komi 7.5` | Sets the komi value in the JSON |
| `play B Q16` | Appends `"Move N: Black plays Q16"` to the moves list |
| `play W D4` | Appends `"Move N: White plays D4"` to the moves list |
| `undo` | Removes the last move from the list |
| `clear_board` | Clears all moves (starts fresh) |
| Everything else | Forwarded to KataGo unchanged, not recorded |

---

## Troubleshooting

### "Lizzie says the engine failed to start"

- Make sure the path in `my_gtp_proxy_cpp.bat` is correct.
- Make sure `gtp_proxy.exe`, `katago.exe`, the weights file, and the config file all exist at the paths specified.
- Try running `my_gtp_proxy_cpp.bat` by double-clicking it. If a window flashes and closes immediately, open a Command Prompt, navigate to the folder, and run the bat manually to see error messages:
  ```
  cd C:\katago_old\lizzie
  my_gtp_proxy_cpp.bat
  ```

### "game_data.json is not being created"

- `game_data.json` is written to the **working directory** of the proxy, which is `C:\katago_old\lizzie\` (set by the `cd /d %~dp0` line in the bat file).
- Make sure you have write permissions to that folder.
- The file is only created after the first `play`, `boardsize`, `komi`, `undo`, or `clear_board` command is received from Lizzie.

### "game_data.json is empty or has no moves"

- If you are only analyzing (not placing stones), no `play` commands are sent by Lizzie, so the moves list will be empty. The file is still updated when `boardsize` or `komi` changes.

### "I want to go back to the old Python proxy"

Change Lizzie's engine command back to:
```
"C:\katago_old\lizzie\my_gtp_proxy.bat"
```

---

## How to compile (if you need to rebuild)

You only need to do this if `gtp_proxy.exe` is missing or you've edited `gtp_proxy.cpp`.

### Prerequisites

- **Visual Studio Build Tools 2019** (or later) with the **C++ desktop workload** installed.

### Steps

1. Open a **Developer Command Prompt for VS** (search for it in the Start menu), or run this in a regular Command Prompt:
   ```
   "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
   ```

2. Navigate to the project folder:
   ```
   cd C:\katago_old\lizzie
   ```

3. Compile:
   ```
   cl /EHsc /std:c++17 /W4 /utf-8 /O2 /I include gtp_proxy.cpp /Fe:gtp_proxy.exe
   ```

4. If successful, you'll see `gtp_proxy.exe` in the folder. Done!

> **Note:** The `include\nlohmann\json.hpp` file must exist. It's the [nlohmann/json](https://github.com/nlohmann/json) single-header library. If it's missing, download it:
> ```
> mkdir include\nlohmann
> curl -L -o include\nlohmann\json.hpp https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp
> ```

---

## File overview

```
C:\katago_old\lizzie\
├── gtp_proxy.cpp              ← C++ source code
├── gtp_proxy.exe              ← compiled proxy (what Lizzie runs)
├── my_gtp_proxy_cpp.bat       ← launch script for Lizzie
├── game_data.json             ← output: live game record (auto-generated)
├── include\
│   └── nlohmann\
│       └── json.hpp           ← JSON library (needed for compilation only)
├── CMakeLists.txt             ← alternative build config (optional)
├── katago.exe                 ← KataGo engine (your existing file)
├── KataGo15b.gz               ← neural network weights (your existing file)
└── default_gtp.cfg            ← KataGo config (your existing file)
```

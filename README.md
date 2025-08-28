Now you have: game records (sgf files)
You want: game analysis of every situation (total number of move in a game = 68 --> 68 situations for that game) in every game from KataGo 18blocks 9x9 in a .jsonl file: 

Pipeline: 
0. https://github.com/lightvector/KataGo/releases/tag/v1.16.3 Go to this website to get all KataGo models, including the KataGo18b9x9.gz package and the katago.exe file that we will use. I would recommend to put all files under the folder where your decompress the downloaded package. 
1. (sgf_to_jsonl.py) extract info from sgf file to a .jsonl file (json_output.jsonl)
2. (analysis.cfg) put this configuration file in the same folder where everything is located
3. copy and paste those line by line to your terminal (ignore comments), wait for a long time, then you will get a .jsonl file (json_output_with_top_k.jsonl), that's what we want: 

# Configuration paths (please change to your own)
$KataGoExe   = "D:\katago_old\lizzie\katago.exe"
$ModelFile   = "D:\katago_old\lizzie\KataGo18b9x9.gz"
$ConfigFile  = "D:\katago_old\lizzie\analysis.cfg"
$InputJsonl  = "D:\katago_old\lizzie\katago_input.jsonl"
$OutputJsonl = "D:\katago_old\lizzie\katago_output.jsonl"

# Check if those files exist (false -->you have  inputed wrong paths)
"exe:       " + (Test-Path $KataGoExe)
"model:     " + (Test-Path $ModelFile)
"config:    " + (Test-Path $ConfigFile)
"inputjson: " + (Test-Path $InputJsonl)

# Run it
Get-Content $InputJsonl -Raw |
  & $KataGoExe analysis `
      -config $ConfigFile `
      -model  $ModelFile `
      -analysis-threads 1 |
  Set-Content $OutputJsonl

---

Noted: 
Ahead of that, I have: 
(1) used total_move_histogram.py to plot "total number of moves of all games records vs frequency"
(2) used pick_random_sgf_based_on_total_move.py to have a glance of the quality (if this game is worth studying for the model) of a random game that has a particular total number of move

Eventually, I decided to preserve games of total number of moves ∈ [20, 70]
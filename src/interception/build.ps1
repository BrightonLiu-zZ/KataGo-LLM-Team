Set-Location "C:\git_repo\KataGo-LLM-Team\src\interception"
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
$output = cmd /c "`"$vcvars`" x64 > nul 2>&1 & cl.exe /std:c++17 /EHsc /utf-8 /O2 gtp_proxy.cpp /Fe:gtp_proxy.exe /link ws2_32.lib 2>&1"
$output
"EXIT: $LASTEXITCODE"

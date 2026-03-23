@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cl.exe /std:c++17 /EHsc /utf-8 /W4 /O2 gtp_proxy_v4.cpp /Fe:gtp_proxy_v4.exe /link ws2_32.lib
echo BUILD EXIT CODE: %ERRORLEVEL%

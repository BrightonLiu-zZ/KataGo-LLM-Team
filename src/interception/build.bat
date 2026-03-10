@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cl.exe /std:c++17 /EHsc /utf-8 /W4 /O2 gtp_proxy.cpp /Fe:gtp_proxy.exe /link ws2_32.lib
echo BUILD EXIT CODE: %ERRORLEVEL%

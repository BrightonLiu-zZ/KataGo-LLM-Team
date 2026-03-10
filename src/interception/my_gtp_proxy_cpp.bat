@echo off
cd /c %~dp0
gtp_proxy.exe C:\katago_old\lizzie\katago.exe gtp -model C:\katago_old\lizzie\KataGo15b.gz -config C:\git_repo\KataGo-LLM-Team\src\interception\python\default_gtp.cfg

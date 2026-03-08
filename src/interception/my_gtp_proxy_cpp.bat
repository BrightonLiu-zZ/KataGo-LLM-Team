@echo off
cd /c %~dp0
gtp_proxy.exe C:\katago_old\lizzie\katago.exe gtp -model C:\katago_old\lizzie\KataGo15b.gz -config C:\katago_old\lizzie\default_gtp.cfg

@echo off
cd /d %~dp0
python D:\katago_old\lizzie\my_gtp_proxy.py D:\katago_old\lizzie\katago.exe gtp -model D:\katago_old\lizzie\KataGo15b.gz -config D:\katago_old\lizzie\default_gtp.cfg

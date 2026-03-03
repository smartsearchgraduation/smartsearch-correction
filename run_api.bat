@echo off
REM Scriptin bulundugu dizine gec
cd /d "%~dp0"

REM Sanal ortami aktif et
call venv\Scripts\activate

REM api.py calculistir
echo api.py calistiriliyor...
python api.py

pause

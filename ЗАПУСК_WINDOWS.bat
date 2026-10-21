@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (set "PY=py -3") else (set "PY=python")
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
  echo Ошибка установки библиотек. Проверьте Python и подключение к интернету.
  pause
  exit /b 1
)
%PY% main.py
if errorlevel 1 pause

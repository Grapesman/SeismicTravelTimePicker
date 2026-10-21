@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (set "PY=py -3") else (set "PY=python")
%PY% -m pip install -r requirements-dev.txt
if errorlevel 1 goto error
%PY% -m pytest -q
if errorlevel 1 goto error
%PY% -m PyInstaller --clean --noconfirm --onefile --windowed --name SeismicTravelTimePicker main.py
if errorlevel 1 goto error
echo Готово: dist\SeismicTravelTimePicker.exe
pause
exit /b 0
:error
echo Сборка завершилась ошибкой. Смотрите вывод выше.
pause
exit /b 1

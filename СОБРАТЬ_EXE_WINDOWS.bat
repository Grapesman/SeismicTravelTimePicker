@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (set "PY=py -3") else (set "PY=python")
%PY% -m pip install -r requirements-dev.txt
if errorlevel 1 goto error
%PY% -m pytest -q
if errorlevel 1 goto error
%PY% -m PyInstaller --clean --noconfirm --onefile --windowed --name SeismicTravelTimePicker --icon assets/seispeak_icon.ico --add-data "assets/seispeak_icon.png;assets" main.py
if errorlevel 1 goto error
echo Готово: dist\SeismicTravelTimePicker.exe
pause
exit /b 0
:error
echo Сборка завершилась ошибкой. Смотрите вывод выше.
pause
exit /b 1

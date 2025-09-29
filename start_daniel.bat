@echo off
echo ===============================
echo  Starting Whisper + Daniel AI
echo ===============================

:: Step 1: Start Whisper server
echo [1/2] Launching Whisper server...
start "" /MIN cmd /c "M:\daniel\venv-whisper\Scripts\activate && python M:\daniel\whisper_server.py"


:: Wait a few seconds for Whisper to spin up
echo Waiting for Whisper to be ready...
timeout /t 5 >nul

:: Step 2: Start Daniel in main venv (same window)
echo [2/2] Launching Daniel AI...
call M:\daniel\venv\Scripts\activate.bat
python M:\daniel\daniel.py

:: Keep window open
pause

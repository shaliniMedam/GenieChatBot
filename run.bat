@echo off
echo Installing dependencies (logging to install.log)...
pip install -r backend\requirements.txt > install.log 2>&1
echo.
echo Starting the Genie backend server...
echo Logging output to server.log so the AI can debug it. Please leave this window open!

start http://localhost:8000/app

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1
pause

@echo off
echo Starting RL Trader System...

REM Start Python script (ZMQ server)
start /B python C:\Users\ingle\OneDrive\Desktop\Actor_Critic_ML_NT\actor_critic.py

REM Wait for model to initialize
echo Waiting for RL model to initialize (5 seconds)...
timeout /t 5 /nobreak

REM Start NinjaTrader
echo Starting NinjaTrader...
start "" "C:\Program Files\NinjaTrader 8\bin\NinjaTrader.exe" --open=Strategy --strategy=RLTrader

echo System startup complete.

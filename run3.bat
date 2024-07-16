@echo off
setlocal

REM Check for the correct number of arguments
if "%~2"=="" (
    echo Usage: run.bat number_clients DataDirectory
    exit /b 1
)

REM Assign the arguments to variables
set number_clients=%1
set DataDirectory=%2

REM Function to start server and clients and wait for completion
:RUN_SERVER_AND_CLIENTS
REM Activate the flowerTutorial environment for the data split step
call conda activate C:\Users\Omar\anaconda3\envs\flowr6

REM Run dataSetspliter.py with the given arguments
python dataSetspliter.py %number_clients% "%~2"

REM Start the server in a separate command window
start "SERVER" cmd /c "conda activate C:\Users\Omar\anaconda3\envs\flowr6 && python server.py %number_clients%"

REM Array to store client process IDs
set "client_pids="

REM Start each client in a separate command window and store process IDs
for /L %%i in (1,1,%number_clients%) do (
    set "file_path=Data\split_%%i.csv"
    start "Client %%i" cmd /c "conda activate flowerTutorial && python client.py %%i Data\split_%%i.csv"

    REM Wait briefly to capture the PID of the client process
    timeout /t 2 /nobreak

    REM Capture the PID of each client process and store it in the array
    for /f "tokens=2" %%a in ('tasklist /v ^| findstr /i "Client %%i"') do (
        set "client_pids=!client_pids! %%a"
    )
)

REM Wait for all clients to finish
:WAIT_FOR_CLIENTS
set "all_clients_finished=true"

for %%p in (%client_pids%) do (
    tasklist /FI "PID eq %%p" | find /i "python.exe" > nul
    if not errorlevel 1 (
        set "all_clients_finished=false"
    )
)

REM If any client is still running, wait and check again
if "%all_clients_finished%"=="false" (
    timeout /t 5 /nobreak
    goto WAIT_FOR_CLIENTS
)

REM Close server and client processes
taskkill /F /PID %client_pids%
taskkill /F /FI "WINDOWTITLE eq SERVER"

REM Optional: Pause between datasets
timeout /t 10 /nobreak

REM End of loop
goto :EOF

:end
endlocal

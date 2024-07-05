@echo off
setlocal

REM Check for the correct number of arguments
if "%~2"=="" (
    echo Usage: run.bat number_clients DataPath
    exit /b 1
)

REM Assign the arguments to variables
set number_clients=%1
set DataPath=%2

REM Activate the flowerTutorial environment for the data split step
call conda activate flower

REM Run dataSetspliter.py with the given arguments
python dataSetspliter.py %number_clients% %DataPath%
timeout /t 2 /nobreak


@REM if errorlevel 1 (
@REM     echo Error running dataSetspliter.py
@REM     exit /b 1
@REM )
REM Run the server.py script in a new command prompt and close it after it is done

@echo off
setlocal
#function that get PID and kill it



REM Start the server and get its PID
start "SERVER" cmd /k "python server.py %number_clients% & taskkill /f /im cmd.exe"
timeout /t 5 /nobreak

for /f "tokens=2 " %%a in ('tasklist /fi "WindowTitle eq SERVER*"') do (
    set "server_pid=%%a"
)
@echo on
echo PID  %server_pid%
@echo off


REM Loop through the number of clients and run client.py for each one in a new command prompt, activating the flowerTutorial environment
for /L %%i in (1,1,%number_clients%) do (
    set "file_path=Data\split_%%i.csv"
    echo python client.py %%i %file_path%
    call conda activate flower
    start "Client %%i" cmd /k "python client.py %%i Data\split_%%i.csv"
)


:loop
tasklist | find " %server_pid% " >nul
if not errorlevel 1  (
    timeout /t 10 >nul
    @echo on
    echo Server is still running
    @echo off
    goto :loop
)

endlocal

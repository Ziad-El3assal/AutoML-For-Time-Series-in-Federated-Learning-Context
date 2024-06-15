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
call conda activate C:\Users\Omar\anaconda3\envs\flowr6

REM Run dataSetspliter.py with the given arguments
python dataSetspliter.py %number_clients% %DataPath%
timeout /t 15 /nobreak


@REM if errorlevel 1 (
@REM     echo Error running dataSetspliter.py
@REM     exit /b 1
@REM )

start "SERVER" cmd /k python server.py %number_clients%
@REM timeout /t 10 /nobreak


REM Loop through the number of clients and run client.py for each one in a new command prompt, activating the flowerTutorial environment
for /L %%i in (1,1,%number_clients%) do (
    set "file_path=Data\split_%%i.csv"
    echo python client.py %%i %file_path%
    call conda activate flowerTutorial
    start "Client %%i" cmd /k "python client.py %%i Data\split_%%i.csv"
)

endlocal

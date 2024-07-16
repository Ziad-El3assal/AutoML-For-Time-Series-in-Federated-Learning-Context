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

REM Loop through all CSV files in the DataDirectory
for %%f in (%DataDirectory%\*.csv) do (
    REM Activate the flowerTutorial environment for the data split step
    call conda activate C:\Users\Omar\anaconda3\envs\flowr6

    REM Run dataSetspliter.py with the given arguments
    python dataSetspliter.py %number_clients% %%f
    @REM timeout /t 15 /nobreak

    @REM REM Check for error in dataSetspliter.py
    @REM if errorlevel 1 (
    @REM     echo Error running dataSetspliter.py on dataset %%f
    @REM     exit /b 1
    @REM )

    REM Start the server and wait for it to complete
    start /wait "SERVER" cmd /c "conda activate C:\Users\Omar\anaconda3\envs\flowr6 && python server.py %number_clients%"

    REM Loop through the number of clients and run client.py for each one, waiting for each to complete
    for /L %%i in (1,1,%number_clients%) do (
        set "file_path=Data\split_%%i.csv"
        echo python client.py %%i %file_path%
        start /wait "Client %%i" cmd /c "conda activate flowerTutorial && python client.py %%i Data\split_%%i.csv"
    )

    @REM REM Optional: Pause between datasets
    @REM timeout /t 15 /nobreak

    @REM REM Wait for a moment before proceeding to the next dataset
    @REM timeout /t 15 /nobreak
)

endlocal

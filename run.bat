@echo off
setlocal

:: Check if the range is provided as an argument
if "%~1"=="" (
    echo Usage: %~nx0 ^<range^>
    exit /b 1
)

:: Check if the path is provided as an argument
if "%~2"=="" (
    echo Usage: %~nx0 ^<range^> ^<path^>
    exit /b 1
)
::run.bat 2 DataSplits\1_splits\10_splits

:: Activate the virtual environment




set "RANGE=%~1"
:: Get the path from the second argument
set "PATH=%~2"
echo Spletting the data set...
start python dataSetSplitter.py %PATH% %RANGE%

:: Start the server.py script
echo Starting server.py...
start python server.py
:: Save the PID of the server
set "SERVER_PID=%!" 


:: Give the server some time to start
timeout /t 1

::DataSplits\1_splits\10_splits
:: Run client{i}.py scripts in a loop
for /L %%i in (1,1,%RANGE%) do (
    echo Starting 
    echo Starting client%%i.py...
    start cmd /k "python client.py ..\Data\split_%i%.csv %%i"
)

:: Wait for all background jobs to finish
echo All client scripts have been started.




@echo off
setlocal

rem Get build name
for /f "usebackq delims=" %%i in (`python .\src\optboolnet\version.py name`) do set "Build=%%i"
set "cwd=%cd%"

echo Uploading %Build% to Anaconda.org

if "%~1"=="" (
    echo Usage: %~nx0 {personal^|group}
    exit /b 1
)

set "CHANNEL=%~1"

if /i "%CHANNEL%"=="personal" (
    echo Uploading optboolnet to your personal Anaconda.org account
    anaconda upload "%cwd%\dist\conda\noarch\%Build%-py_0.conda"
) else if /i "%CHANNEL%"=="group" (
    echo Uploading optboolnet to Anaconda.org group 'msolab'
    anaconda upload --user msolab "%cwd%\dist\conda\noarch\%Build%-py_0.conda"
) else (
    echo Unknown option: %CHANNEL% (must be 'personal' or 'group')
    exit /b 1
)

endlocal
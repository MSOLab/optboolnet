if not exist ".\dist" md ".\dist"
python .\src\optboolnet\version.py name > temp_output
set /p Build=<temp_output
del temp_output

echo Building %Build% for Anaconda.org

rem default: run tests
set "TEST_ARGS="
if /i "%~1"=="--skip-test" (
    echo Skipping tests during conda-build
    set "TEST_ARGS=--no-test"
)

conda-build ".\recipe" --output-folder ".\dist\conda" -c colomoto -c gurobi -c conda-forge %TEST_ARGS%
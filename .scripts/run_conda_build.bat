if not exist ".\dist" md ".\dist"
python .\src\optboolnet\version.py name > temp_output
set /p Build=<temp_output
DEL temp_output
echo "Building %Build% for Anaconda.org"
conda build ".\recipe" --output-folder ".\dist\conda" -c colomoto -c gurobi -c conda-forge
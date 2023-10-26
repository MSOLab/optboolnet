python .\src\optboolnet\version.py name > temp_output
set /p Build=<temp_output
DEL temp_output
echo "Building %Build%"
python -m build -s -w
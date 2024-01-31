python .\src\optboolnet\version.py name > temp_output
set /p Build=<temp_output
DEL temp_output
echo "Building %Build% for PyPI"
python -m build -s -w
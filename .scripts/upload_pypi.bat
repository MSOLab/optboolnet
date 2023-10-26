python .\src\optboolnet\version.py name > temp_output
set /p Build=<temp_output
DEL temp_output
echo "Uploading %Build% to PyPI"

twine upload %cd%\dist\%Build%.tar.gz 
 
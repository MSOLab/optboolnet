python .\src\optboolnet\version.py name > temp_output
set /p Build=<temp_output
DEL temp_output
echo "Uploading %Build% to Anaconda.org"
anaconda upload %cd%\dist\conda\noarch\%Build%-py_0.tar.bz2
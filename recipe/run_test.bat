
pip install colomoto-jupyter pyomo gurobipy pytest

pip check

pytest .\tests

IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
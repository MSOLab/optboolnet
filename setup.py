import setuptools
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
data = dict()
exec(open("src/optboolnet/version.py").read(), data)

setuptools.setup(version=data["__version__"])

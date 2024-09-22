from setuptools import setup, Extension
import pybind11
# python3 -m pybind11 --includes
functions_module = Extension(  
    name ='NPmetric',  
    sources = ['solver.cpp', '../MyUtils/vectorND.cpp', '../MyUtils/ufs.cpp', 'scanLine.cpp', 'grouping.cpp', 'timeAlign.cpp', '../Bundlemetric/NPmetric.cpp'],
    include_dirs = [pybind11.get_include(), '../', '../box2d/include/', '../MyUtils/', '../Bundlemetric/'],
    extra_compile_args=["-fopenmp", "-O3", "-std=c++17"],
    extra_link_args=[],
    language='c++',
)  
  
setup(ext_modules = [functions_module])

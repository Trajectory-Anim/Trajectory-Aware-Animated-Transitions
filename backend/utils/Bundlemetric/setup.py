from setuptools import setup, Extension
import pybind11
# python3 -m pybind11 --includes
functions_module = Extension(  
    name ='Bundlemetric',  
    sources = ['solver.cpp', '../MyUtils/vectorND.cpp', 'NPmetric.cpp', 'Qualitymetric.cpp', '../MyUtils/ufs.cpp'],
    include_dirs = [pybind11.get_include(), '../', '../box2d/include/', '../MyUtils/', '../Bundlemetric/'],
    extra_compile_args=["-fopenmp", "-O3", "-std=c++17"],
    extra_link_args=[],
    language='c++',
)  
  
setup(ext_modules = [functions_module])
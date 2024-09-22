from setuptools import setup, Extension
import pybind11
# python3 -m pybind11 --includes
functions_module = Extension(  
    name ='pdutils',
    sources = ['solver.cpp', '../MyUtils/vectorND.cpp', 'pdutils.cpp'],
    include_dirs = [pybind11.get_include(), '../', '../box2d/include/', '../MyUtils/'],
    extra_compile_args=["/openmp", "/O2", "/std:c++17"],
    extra_link_args=[],
    language='c++',
)
  
setup(ext_modules = [functions_module])

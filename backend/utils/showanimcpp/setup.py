from setuptools import setup, Extension
import pybind11
# python3 -m pybind11 --includes
functions_module = Extension(  
    name ='showanimcpp',    
    sources = ['solver.cpp', '../MyUtils/vectorND.cpp', 'showanimcpp.cpp'],
    include_dirs = [pybind11.get_include(), '../', '../MyUtils/', '../opencv2/include/'],
    library_dirs= ['../opencv2/lib'],
    libraries= ['opencv_world490'],
    extra_compile_args=["-fopenmp", "-O3", "-std=c++17"],
    extra_link_args=[],
    language='c++',
)  
  
setup(ext_modules = [functions_module])

from setuptools import setup, Extension
import pybind11
import glob

cpp_files = glob.glob('../box2d/src/collision/*.cpp') + glob.glob('../box2d/src/common/*.cpp') + glob.glob('../box2d/src/dynamics/*.cpp') + glob.glob('../box2d/src/rope/*.cpp')

# python3 -m pybind11 --includes
functions_module = Extension(  
    name ='Animsolver',  
    sources = ['solver.cpp', '../MyUtils/vectorND.cpp', '../MyUtils/ufs.cpp', 'anim_compute.cpp', 'circle_b2d.cpp'] + cpp_files,
    # include_dirs = [pybind11.get_include(), '../', './include/', '../MyUtils/','C:/Users/DuanLi/Downloads/opencv/build/include'],
    include_dirs = [pybind11.get_include(), '../', './include/', '../MyUtils/', '../box2d/include/', '../opencv2/include/'],
    library_dirs=['../opencv2/lib'],
    libraries=['opencv_world490'],
    extra_compile_args=["/openmp", "/O2", "/std:c++17"],
    # extra_link_args=['C:/Users/DuanLi/Downloads/opencv/build/x64/vc15/lib/opencv_world3414.lib'],
    extra_link_args=[],
    language='c++',
)
  
setup(ext_modules = [functions_module])

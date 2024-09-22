#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ctime>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include "NPmetric.h"
#include "Qualitymetric.h"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(Bundlemetric, m) {
    m.doc() = "Bundle metric calculator";  // optional module docstring
    m.def("get_path_distance", &get_path_distance, "A function");
    m.def("get_SDTW", &get_SDTW, "A function");
    m.def("get_DTW", &get_DTW, "A function");
    m.def("get_NP_global", &get_NP_global, "A function");
    m.def("get_NP_local", &get_NP_local, "A function");
    m.def("get_ink_ratio", &get_ink_ratio, "A function");
    m.def("get_distortion", &get_distortion, "A function");
    m.def("get_split_num", &get_split_num, "A function");
}

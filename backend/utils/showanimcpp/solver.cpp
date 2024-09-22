#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ctime>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include "showanimcpp.h"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(showanimcpp, m) {
    m.doc() = "Show Animation C++";  // optional module docstring
    m.def("show_animation", &show_animation, "A function");
}

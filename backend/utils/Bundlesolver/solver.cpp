#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ctime>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include "forceDirected.h"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(Bundlesolver, m) {
    m.doc() = "Edge Bundling Solver";  // optional module docstring
    m.def("get_control_points_lists", &get_control_points_lists, "A function");
}

#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ctime>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include "pdutils.h"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(pdutils, m) {
    m.doc() = "Power diagram Utils";  // optional module docstring
    m.def("find_circle", &find_circle, "A function");
    m.def("interpolate", &interpolate, "A function");
    m.def("multi_interpolate", &multi_interpolate, "A function");
    m.def("cell_centroid", &cell_centroid, "A function");
    m.def("inscribed_circle_radius", &inscribed_circle_radius, "A function");
    // m.def("interpolate", &interpolate, "A function");
    // m.def("get_NP_local", &get_NP_local, "A function");
}

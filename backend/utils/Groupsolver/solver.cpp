#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ctime>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include "grouping.h"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(Groupsolver, m) {
    m.doc() = "Grouping Solver";  // optional module docstring
    m.def("get_group", &get_group, "A function");
}

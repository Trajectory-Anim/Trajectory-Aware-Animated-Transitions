#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ctime>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include "anim_compute.h"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(Animsolver, m) {
    m.doc() = "Animation Solver";  // optional module docstring
    m.def("anim_compute", &anim_compute, "A function");
}

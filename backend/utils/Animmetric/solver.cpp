#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ctime>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include "Animmetric.h"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(Animmetric, m) {
    m.doc() = "Animation metric calculator";  // optional module docstring
    m.def("OuterOcclusion", &OuterOcclusion, "A function");
    m.def("WithinGroupOcclusion", &WithinGroupOcclusion, "A function");
    m.def("OverallOcclusion", &OverallOcclusion, "A function");
    m.def("Dispersion", &Dispersion, "A function");
    m.def("Deformation", &Deformation, "A function");
}

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vector>

#include "vectorND.h"

using namespace std;
namespace py = pybind11;

vector<vector<int>> get_group(const py::object& anim, vector3D, int);

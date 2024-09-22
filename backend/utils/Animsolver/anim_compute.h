#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vector>

#include "vectorND.h"

using namespace std;
namespace py = pybind11;

tuple<vector3D,
      vector3D,
      vector<int>,
      vector<int>>
anim_compute(const py::object& anim);

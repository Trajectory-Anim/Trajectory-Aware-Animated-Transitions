#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "vectorND.h"

using namespace std;
namespace py = pybind11;

double get_ink_ratio(const vector3D& init_paths, const vector3D& bundled_paths);

double get_distortion(const vector3D& init_paths, const vector3D& bundled_paths);

double get_split_num(const vector3D& bundled_paths);

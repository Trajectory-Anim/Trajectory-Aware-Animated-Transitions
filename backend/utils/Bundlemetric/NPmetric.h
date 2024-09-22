#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "vectorND.h"

using namespace std;
namespace py = pybind11;

double get_path_overlap(const vector2D& path1, const vector2D& path2);
vector2D get_all_path_overlap(const vector3D& paths);
double get_SDTW(const vector2D& path1, const vector2D& path2);
double get_DTW(const vector2D& path1, const vector2D& path2);
double get_path_distance(const vector2D& path1, const vector2D& path2);
double get_NP_global(const vector3D& init_paths, const vector3D& bundled_paths, int k);
double get_NP_local(const vector3D& init_paths, const vector3D& bundled_paths, const vector2D& passing_flags, int k, int bundle_legacy);

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "vectorND.h"

using namespace std;
namespace py = pybind11;

// double get_path_overlap(const vector2D& path1, const vector2D& path2);
tuple<double, double, double> find_circle(vector2D& positions);
void min_circle_cover(vector2D pos, vector1D& center, double& radius);
void circle_center3(vector1D p0, vector1D p1, vector1D p2, vector1D& cp);
bool point_in(vector1D p, vector1D c, double r);

vector2D interpolate(double center_x, double center_y, double radius, int pcs=360);
vector2D multi_interpolate(const vector2D& centers, const vector1D& radii);
tuple<double,double> cell_centroid(const vector2D& cell);
double inscribed_circle_radius(const vector2D& cell, const vector1D& site);
// double get_path_distance(const vector2D& path1, const vector2D& path2);
// double get_NP_global(const vector3D& init_paths, const vector3D& bundled_paths, int k);
// double get_NP_local(const vector3D& init_paths, const vector3D& bundled_paths, const vector2D& passing_flags, int k, int bundle_legacy);

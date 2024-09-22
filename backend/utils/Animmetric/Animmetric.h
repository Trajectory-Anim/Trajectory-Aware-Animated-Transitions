#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "vectorND.h"

using namespace std;
namespace py = pybind11;

vector1D OuterOcclusion(const vector3D& anime_position, const vector<vector<int>>& anime_groups, const vector1D& radii, vector<int>& starts, vector<int>& ends);
vector2D WithinGroupOcclusion(const vector3D& anime_position, const vector<vector<int>>& anime_groups, const vector1D& radii, vector<int>& starts, vector<int>& ends);
vector1D OverallOcclusion(const vector3D& anime_position, const vector<vector<int>>& anime_groups, const vector1D& radii, vector<int>& starts, vector<int>& ends);
vector2D Dispersion(const vector3D& anime_position, const vector<vector<int>>& anime_groups, const vector1D& radii, vector<int>& starts, vector<int>& ends);
vector2D Deformation(const vector3D& anime_position, const vector<vector<int>>& anime_groups, const vector1D& radii, vector<int>& starts, vector<int>& ends);

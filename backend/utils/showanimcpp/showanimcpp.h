#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "vectorND.h"

using namespace std;
namespace py = pybind11;

void show_animation(vector3D anime_position,
                    vector<vector<int>> cluster,
                    vector<int> anime_start,
                    vector<int> anime_end,
                    vector1D radii,
                    int width = 1024,
                    int height = 1024,
                    string title = "animation",
                    int fps = 30,
                    double margin_percentage = 0.05,
                    double fade_time = 0.5,
                    string save_to_filename = "animation.mp4",
                    int resolution_scale = 1,
                    vector3D bundled_position = vector3D(),
                    bool trace_mode = true,
                    vector<int> points_to_trace = vector<int>(),
                    int grid_n = 1);

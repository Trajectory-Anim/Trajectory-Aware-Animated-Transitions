#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ctime>
#include <iostream>
#include <utility>
#include <vector>
#include <chrono>

#include "NPmetric.h"
#include "grouping.h"
#include "ufs.h"
#include "vectorND.h"
#define THREADS_NUM 48

using namespace std;
namespace py = pybind11;

vector2D remove_repeat_path_points(const vector2D& path) {
    vector2D newPath;
    newPath.push_back(path[0]);
    int N_path = (int) path.size();
    for (int i = 1; i < N_path; i++) {
        if (get_norm(path[i] - path[i - 1]) > 1e-5) {
            newPath.push_back(path[i]);
        }
    }
    return newPath;
}


vector<vector<int>> get_group(const py::object& anim, vector3D control_points, int stage) {
    cout << "  Computing distance matrix..." << endl;

    int N = (int) control_points.size();
    double n_c = anim.attr("n_c").cast<double>();

    vector2D distance_matrix = get_zero2D(N, N);

    auto time_1 = chrono::steady_clock::now();

    vector3D new_paths(N);
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        new_paths[i] =  remove_repeat_path_points(control_points[i]);
    }


#pragma omp parallel for num_threads(THREADS_NUM)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                distance_matrix[i][j] = 0;
            } else {
                distance_matrix[i][j] = 1 - get_path_overlap(new_paths[i], new_paths[j]);
            } 
        }
    }

    auto time_2 = chrono::steady_clock::now();

    double interface_dis_th = 1 - n_c;

    py::object interface_module =
        py::module::import("utils.Groupsolver.agg_interface");
    py::object interface = interface_module.attr("get_subgroup_result");

    cout << "  Computing agglomerative clustering..." << endl;

    vector<vector<int>> group_indicator = interface(distance_matrix, interface_dis_th, stage).cast<vector<vector<int>>>();

    auto time_3 = chrono::steady_clock::now();

    double time_span1 = chrono::duration_cast<chrono::duration<double>>(time_2 - time_1).count();
    double time_span2 = chrono::duration_cast<chrono::duration<double>>(time_3 - time_2).count();
    // cout << "  Computing distance matrix time : " << time_span1 << "s" << endl;
    // cout << "  Computing agglomerative clustering time : " << time_span2 << "s" << endl;
    return group_indicator;
}

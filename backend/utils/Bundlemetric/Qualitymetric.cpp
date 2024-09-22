#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_map>

#include "Qualitymetric.h"
#include "vectorND.h"
#include "ufs.h"

using namespace std;
namespace py = pybind11;

double get_path_length(const vector2D& path) {
    int N_path = (int)path.size();
    double result = 0;
    for (int i = 0; i < N_path - 1; i++) {
        result += get_norm(path[i] - path[i + 1]);
    }
    return result;
}

double get_ink_ratio(const vector3D& init_paths, const vector3D& bundled_paths) {
    int N = (int)init_paths.size();
    double init_ink = 0;

    #pragma omp parallel for reduction(+:init_ink)
    for (int i = 0; i < N; i++) {
        double cur_path_length = get_path_length(init_paths[i]);
        init_ink += cur_path_length;
    }

    double bundled_ink = 0;
    // compute the bundled ink

    int N_all_CP = 0;
    vector2D pos;
    vector<vector<int>> id_map(N);
    for (int i = 0; i < N; i++) {
        int N_CP = (int)bundled_paths[i].size();
        for (int j = 0; j < N_CP; j++) {
            id_map[i].push_back(N_all_CP);
            pos.push_back(bundled_paths[i][j]);
            N_all_CP++;
        }
    }

    // merge overlap control point
    UFS ufs(N_all_CP);
    for (int i = 0; i < N_all_CP; i++) {
        for (int j = 0; j < N_all_CP; j++) {
            if (get_norm(pos[i] - pos[j]) < 1e-5) {
                ufs.union_set(i, j);
            }
        }
    }

    for (int i = 0; i < N_all_CP; i++) {
        ufs.find(i);
    }

    map<tuple<int, int>, bool> visit;
    
    // compute the bundled ink
    for (int i = 0; i < N; i++) {
        int N_CP = (int)bundled_paths[i].size();
        for (int j = 0; j < N_CP - 1; j++) {
            int id1 = ufs.find(id_map[i][j]);
            int id2 = ufs.find(id_map[i][j + 1]);
            if (id1 == id2) {
                continue;
            } else if (visit.count({id1, id2}) > 0) {
                continue;
            } else {
                visit[{id1, id2}] = true;
                bundled_ink += get_norm(pos[id1] - pos[id2]);
            }
        }
    }

    return bundled_ink / init_ink;
}

double get_split_num(const vector3D& bundled_paths) {
    int N = (int)bundled_paths.size();
    // cout << "N: " << N << endl;
    int N_all_CP = 0;
    vector2D pos;
    vector<vector<int>> id_map(N);
    for (int i = 0; i < N; i++) {
        int N_CP = (int)bundled_paths[i].size();
        for (int j = 0; j < N_CP; j++) {
            id_map[i].push_back(N_all_CP);
            pos.push_back(bundled_paths[i][j]);
            N_all_CP++;
        }
    }

    // cout << "N_ALL_CP: " << N_all_CP << endl;

    // merge overlap control point
    UFS ufs(N_all_CP);
    for (int i = 0; i < N_all_CP; i++) {
        for (int j = 0; j < N_all_CP; j++) {
            if (get_norm(pos[i] - pos[j]) < 1e-5) {
                ufs.union_set(i, j);
            }
        }
    }

    for (int i = 0; i < N_all_CP; i++) {
        ufs.find(i);
    }

    map<tuple<int, int>, bool> visit;
    vector<int> deg(N_all_CP, 0);
    
    // compute the bundled ink
    for (int i = 0; i < N; i++) {
        int N_CP = (int)bundled_paths[i].size();
        for (int j = 0; j < N_CP - 1; j++) {
            int id1 = ufs.find(id_map[i][j]);
            int id2 = ufs.find(id_map[i][j + 1]);
            if (id1 == id2) {
                continue;
            } else if (visit.count({id1, id2}) > 0) {
                continue;
            } else {
                visit[{id1, id2}] = true;
                deg[id1]++;
                deg[id2]++;
            }
        }
    }

    // for (int i = 0; i < N_all_CP; i++) {
    //     cout << "deg[i] : " << deg[i] << endl;
    // }

    int split_num = 0;
    #pragma omp parallel for reduction(+:split_num)
    for (int i = 0; i < N_all_CP; i++) {
        if (deg[i] > 2) {
            split_num ++;
        }
    }

    return split_num;
}

double get_distortion(const vector3D& init_paths, const vector3D& bundled_paths) {
    int N = (int)init_paths.size();

    double distortion = 0;
    #pragma omp parallel for reduction(+:distortion)
    for (int i = 0; i < N; i++) {
        double init_path_length = get_path_length(init_paths[i]);
        double bundled_path_length = get_path_length(bundled_paths[i]);
        double rate1 = bundled_path_length / init_path_length;
        double rate2 = init_path_length / bundled_path_length;
        distortion += max(rate1, rate2);
    }

    return distortion / (double)N;
}

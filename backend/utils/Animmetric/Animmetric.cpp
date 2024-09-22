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

#include "Animmetric.h"
#include "vectorND.h"

using namespace std;
namespace py = pybind11;

vector1D OuterOcclusion(const vector3D& anime_position, const vector<vector<int>>& anime_groups, const vector1D& radii, vector<int>& starts, vector<int>& ends){
    int T = anime_position.size();
    int N = anime_position[0].size();
    int k = anime_groups[0].size();
    vector2D occlusion_per_time_group = get_zero2D(T, k);

    for (int clusteridx = 0; clusteridx < k; clusteridx++){
        vector<int> group;
        for (int point = 0; point < N; point++){
            if (anime_groups[point][clusteridx] == 1){
                group.push_back(point);
            }
        }
        if (group.size() == N){
            continue;
        }
        for (int p : group){
            for (int q = 0; q < N; q++){
                if (find(group.begin(), group.end(), q) != group.end()){
                    continue;
                }
                for (int t = 0; t < T; t++){
                    // cout<<"t: "<<t<<", starts[q]: "<<starts[q]<<", ends[q]: "<<ends[q]<<", starts[p]: "<<starts[p]<<", ends[p]: "<<ends[p]<<endl;
                    if (t <= starts[q] || t >= ends[q] || t<=starts[p] || t>=ends[p]){
                        continue;
                    }
                    if (get_norm(anime_position[t][p] - anime_position[t][q]) < radii[p] + radii[q] - 1e-3){
                        occlusion_per_time_group[t][clusteridx] += 1;
                        // cout<<"+1"<<endl;
                    }
                }
            }
        }
        
        for (int t = 0; t < T; t++){
            occlusion_per_time_group[t][clusteridx] /= (group.size() * (N - group.size()));
        }
    }

    vector1D occlusion_per_time(T, 0);

    for (int t = 0; t < T; t++){
        for (int clusteridx = 0; clusteridx < k; clusteridx++){
            occlusion_per_time[t] += occlusion_per_time_group[t][clusteridx];
        }
        occlusion_per_time[t] /= k;
    }
    return occlusion_per_time;
}
vector2D WithinGroupOcclusion(const vector3D& anime_position, const vector<vector<int>>& anime_groups, const vector1D& radii, vector<int>& starts, vector<int>& ends){
    int T = anime_position.size();
    int N = anime_position[0].size();
    int k = anime_groups[0].size();
    vector2D occlusion_per_time_group = get_zero2D(T, k);

    for (int clusteridx = 0; clusteridx < k; clusteridx++){
        vector<int> group;
        for (int point = 0; point < N; point++){
            if (anime_groups[point][clusteridx] == 1){
                group.push_back(point);
            }
        }
        if (group.size() == 1){
            continue;
        }
        for (int p : group){
            for (int q : group){
                if (p == q){
                    continue;
                }
                for (int t = 0; t < T; t++){
                    if (t <= starts[q] || t >= ends[q] || t<=starts[p] || t>=ends[p]){
                        continue;
                    }
                    if (get_norm(anime_position[t][p] - anime_position[t][q]) < radii[p] + radii[q] - 1e-3){
                         occlusion_per_time_group[t][clusteridx] += 1;
                    }
                }
            }
        }

        for (int t = 0; t < T; t++){
            occlusion_per_time_group[t][clusteridx] /= (group.size() * (group.size() - 1));
        }
    }

    return occlusion_per_time_group;

    // vector1D occlusion_per_time(T, 0);

    // for (int t = 0; t < T; t++){
    //     for (int clusteridx = 0; clusteridx < k; clusteridx++){
    //         occlusion_per_time[t] += occlusion_per_time_group[t][clusteridx];
    //     }
    //     occlusion_per_time[t] /= k;
    // }
    // return occlusion_per_time;
}
vector1D OverallOcclusion(const vector3D& anime_position, const vector<vector<int>>& anime_groups, const vector1D& radii, vector<int>& starts, vector<int>& ends){
    int T = anime_position.size();
    int N = anime_position[0].size();
    
    vector1D occlusion_per_time(T, 0);

    if (N == 1){
        return occlusion_per_time;
    }

    for (int p = 0; p < N; p++){
        for (int q = 0; q < N; q++){
            if (p == q){
                continue;
            }
            for (int t = 0; t < T; t++){
                if (t <= starts[q] || t >= ends[q] || t<=starts[p] || t>=ends[p]){
                    continue;
                }
                if (get_norm(anime_position[t][p] - anime_position[t][q]) < radii[p] + radii[q] - 1e-3){
                    occlusion_per_time[t] += 1;
                }
            }
        }
    }

    for (int t = 0; t < T; t++){
        occlusion_per_time[t] /= (N * (N - 1));
    }

    return occlusion_per_time;
}
vector2D Dispersion(const vector3D& anime_position, const vector<vector<int>>& anime_groups, const vector1D& radii, vector<int>& starts, vector<int>& ends){
    int T = anime_position.size();
    int N = anime_position[0].size();
    int k = anime_groups[0].size();

    vector2D dispersion_per_time_group = get_zero2D(T, k);

    for (int clusteridx = 0; clusteridx < k; clusteridx++){
        vector<int> group;
        for (int point = 0; point < N; point++){
            if (anime_groups[point][clusteridx] == 1){
                group.push_back(point);
            }
        }
        if (group.size() == 1){
            continue;
        }
        for (int p : group){
            for (int q : group){
                if (p == q){
                    continue;
                }
                for (int t = 0; t < T; t++){
                    if (t <= starts[q] || t >= ends[q] || t<=starts[p] || t>=ends[p]){
                        continue;
                    }
                    dispersion_per_time_group[t][clusteridx] += max(0.0, get_norm(anime_position[t][p] - anime_position[t][q])-radii[p]-radii[q]);
                }
            }
        }

        for (int t = 0; t < T; t++){
            dispersion_per_time_group[t][clusteridx] /= group.size() * (group.size() - 1);
        }
    }

    return dispersion_per_time_group;

    // vector1D dispersion_per_time(T, 0);
    // for (int t = 0; t < T; t++){
    //     for (int clusteridx = 0; clusteridx < k; clusteridx++){
    //         dispersion_per_time[t] += dispersion_per_time_group[t][clusteridx];
    //     }
    //     dispersion_per_time[t] /= k;
    // }

    // return dispersion_per_time;
}
vector2D Deformation(const vector3D& anime_position, const vector<vector<int>>& anime_groups, const vector1D& radii, vector<int>& starts, vector<int>& ends){
    int T = anime_position.size();
    int N = anime_position[0].size();
    int k = anime_groups[0].size();

    vector2D deformation_per_time_group = get_zero2D(T - 1, k);

    for (int clusteridx = 0; clusteridx < k; clusteridx++){
        vector<int> group;
        for (int point = 0; point < N; point++){
            if (anime_groups[point][clusteridx] == 1){
                group.push_back(point);
            }
        }
        if (group.size() == 1){
            continue;
        }
        for (int p : group){
            for (int q : group){
                if (p == q){
                    continue;
                }
                for (int t = 1; t < T; t++){
                    if (t <= starts[q] || t >= ends[q] || t<=starts[p] || t>=ends[p]){
                        continue;
                    }
                    double dis_before = max(0.0 ,get_norm(anime_position[t - 1][p] - anime_position[t - 1][q])-radii[p]-radii[q]);
                    double dis_after = max(0.0 ,get_norm(anime_position[t][p] - anime_position[t][q])-radii[p]-radii[q]);
                    deformation_per_time_group[t - 1][clusteridx] += abs(dis_after - dis_before);
                }
            }
        }

        for (int t = 0; t < T - 1; t++){
            deformation_per_time_group[t][clusteridx] /= group.size() * (group.size() - 1);
        }
    }

    return deformation_per_time_group;

    // vector1D deformation_per_time(T - 1, 0);
    // for (int t = 0; t < T - 1; t++){
    //     for (int clusteridx = 0; clusteridx < k; clusteridx++){
    //         deformation_per_time[t] += deformation_per_time_group[t][clusteridx];
    //     }
    //     deformation_per_time[t] /= k;
    // }
    // return deformation_per_time;
}

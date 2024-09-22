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

#include "NPmetric.h"
#include "vectorND.h"

using namespace std;
namespace py = pybind11;

double point_to_line_distance(const vector1D& point, const vector2D& line) {
    double px = point[0];
    double py = point[1];

    double x1 = line[0][0];
    double y1 = line[0][1];
    double x2 = line[1][0];
    double y2 = line[1][1];

    if (get_norm(point - line[0]) < 1e-5) {
        return 0;
    } else if (get_norm(point - line[1]) < 1e-5) {
        return 0;
    }

    double lineMagnitude = get_norm(line[0] - line[1]);
    if (lineMagnitude < 0.00000001) {
        return get_norm(point - line[0]);
    } else {
        double u1 = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1);
        double u = u1 / (lineMagnitude * lineMagnitude);
        if (u < 0.00001 || u > 1) {

            double ix = get_norm(point - line[0]);
            double iy = get_norm(point - line[1]);
            if (ix > iy) {
                return iy;
            } else {
                return ix;
            }
        } else {
            double ix = x1 + u * (x2 - x1);
            double iy = y1 + u * (y2 - y1);
            return get_norm(point - vector1D({ix, iy}));
        }
    }
}

double get_path_overlap(const vector2D& path1, const vector2D& path2) {
    int N_path1 = (int) path1.size();
    int N_path2 = (int) path2.size();

    if (N_path1 <= 2 || N_path2 <= 2) {
        return 0;
    }

    // set a huge cost for the inverse
    vector1D dir_1 = get_unit_vector(path1[N_path1 - 1] - path1[0]);
    vector1D dir_2 = get_unit_vector(path2[N_path2 - 1] - path2[0]);
    if (vector_dot(dir_1, dir_2) < -0.5) {
        return -1e5;
    }

    double overlap_length = 0;
    // bool last_overlap = false;
    double path1_length = 0;
    for (int i = 0; i < N_path1 - 1; i++) {
        path1_length += get_norm(path1[i + 1] - path1[i]);
    }
            
    for (int i = 1; i < N_path1 - 1; i++) {
        bool flag = false;

        // check distance between point and line
        for (int j = 0; j < N_path2 - 1; j++) {
            vector1D point = path1[i];
            vector2D line = vector2D({path2[j], path2[j + 1]});
            auto dir_line = get_unit_vector(line[1] - line[0]);
            auto dir_ref = get_unit_vector(path1[i] - path1[i - 1]);
            if (vector_dot(dir_line, dir_ref) < 0 && get_norm(dir_line) > 1e-5 && get_norm(dir_ref) > 1e-5) {
                continue;
            } 

            double dis_to_line = point_to_line_distance(point, line);

            if (dis_to_line < 1e-5) {
                flag = true;
                break;
            }
        }

                // check distance between point and point
                // for (int j = 1; j < n_control_points_q - 1; j++) {
                //     vector1D pointI = control_points[p][i];
                //     vector1D pointJ = control_points[q][j];
                //     if (get_norm(pointI - pointJ) < 1e-5) {
                //         flag = true;
                //         break;
                //     }
                // }

        if (flag) {
            // if (last_overlap) {
            //     overlap_length += get_norm(path1[i] - path1[i - 1]);
            // }
            
            // compute the control segment length of this control point
            double dis_prev = get_norm(path1[i] - path1[i - 1]);
            double dis_next = get_norm(path1[i + 1] - path1[i]);

            if (i == 1) {
                overlap_length += dis_prev;
            } else {
                overlap_length += dis_prev / 2;
            }

            if (i == N_path1 - 2) {
                overlap_length += dis_next;
            } else {
                overlap_length += dis_next / 2;
            }
        }
        // last_overlap = flag;
    }

    // special case
    // if ((N_path1 == 3 || N_path2 == 3) && last_overlap) {
    //     return 1;
    // }

    return overlap_length / path1_length;
}

vector2D get_all_path_overlap(const vector3D& paths) {
    int N = (int) paths.size();

    // get all paths' length
    vector1D path_lengths(N);
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double length = 0;
        int N_path = (int) paths[i].size();
        for (int j = 0; j < N_path - 1; j++) {
            length += get_norm(paths[i][j + 1] - paths[i][j]);
        }
        path_lengths[i] = length;
    }

    // prepare to sort all control points
    struct PathNode {
        vector1D pos;
        int belong;
        double area;
    };

    vector<PathNode> all_nodes;
    vector<vector<int>> path_node_ids(N);
    int N_all_CP = 0;

    for (int i = 0; i < N; i++) {
        int N_path = (int) paths[i].size();
        for (int j = 1; j < N_path - 1; j++) {
            PathNode node;
            node.pos = paths[i][j];
            node.belong = i;
            node.area = 0;
            if (j == 1) {
                node.area += get_norm(paths[i][j] - paths[i][j - 1]);
            } else {
                node.area += get_norm(paths[i][j] - paths[i][j - 1]) / 2;
            }
            if (j == N_path - 2) {
                node.area += get_norm(paths[i][j] - paths[i][j + 1]);
            } else {
                node.area += get_norm(paths[i][j] - paths[i][j + 1]) / 2;
            }

            all_nodes.push_back(node);
            path_node_ids[i].push_back(N_all_CP++);
        }
    }

    // prepare different directions
    int N_dir = 8;
    vector2D dir_vectors(N_dir);
#pragma omp parallel for
    for (int d = 0; d < N_dir; d++) {
        double angle = (double)d / (double)N_dir * 3.1415;
        dir_vectors[d] = vector1D({cos(angle), sin(angle)});
    }

    // project the control points onto the different directions
    vector2D dir_proj_pos = get_zero2D(N_dir, N_all_CP);
#pragma omp parallel for
    for (int d = 0; d < N_dir; d++) {
        for (int i = 0; i < N_all_CP; i++) {
            dir_proj_pos[d][i] = vector_dot(dir_vectors[d], all_nodes[i].pos);
        }
    }

    // sort the points according to the projection
    vector<vector<int>> dir_order(N_dir);
    vector<vector<int>> dir_rank(N_dir);
#pragma omp parallel for
    for (int d = 0; d < N_dir; d++) {
        for (int i = 0; i < N_all_CP; i++) {
            dir_order[d].push_back(i);
            dir_rank[d].push_back(0);
        }

        sort(dir_order[d].begin(), dir_order[d].end(), [&](int i1, int i2) {
            return dir_proj_pos[d][i1] < dir_proj_pos[d][i2];
        });

        for (int rank = 0; rank < N_all_CP; rank++) {
            dir_rank[d][dir_order[d][rank]] = rank;
        }
    }

    // check overlap
    vector<vector<bool>> overlaped(N_all_CP);
    for (int i = 0; i < N_all_CP; i++) {
        for (int j = 0; j < N; j++) {
            overlaped[i].push_back(false);
        }
    }

    // BUG!!!! need to check all the direction
    double check_eps = 1e-5;
#pragma omp parallel for
    for (int i = 0; i < N_all_CP; i++) {
        for (int d = 0; d < N_dir; d++) {
            int rank = dir_rank[d][i];
            for (int j = rank - 1; j >= 0; j--) {
                int test_id = dir_order[d][j];
                if (abs(dir_proj_pos[d][test_id] - dir_proj_pos[d][i]) > check_eps) {
                    break;
                }
                overlaped[i][all_nodes[test_id].belong] = true;
            }

            for (int j = rank + 1; j < N_all_CP; j++) {
                int test_id = dir_order[d][j];
                if (abs(dir_proj_pos[d][test_id] - dir_proj_pos[d][i]) > check_eps) {
                    break;
                }
                overlaped[i][all_nodes[test_id].belong] = true;
            }
        }
    }

    vector2D result = get_zero2D(N, N);
#pragma omp parallel for
    for (int i = 0; i < N_all_CP; i++) {
        for (int j = 0; j < N; j++) {
            if (overlaped[i][j]) {
                result[all_nodes[i].belong][j] += all_nodes[i].area;
            }
        }
    }
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                result[i][j] = 1;
            } else {
                result[i][j] /= path_lengths[i];
            }
        }
    }
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int N_path_i = (int) paths[i].size();
            int N_path_j = (int) paths[j].size();
            vector1D dir_i = get_unit_vector(paths[i][N_path_i - 1] - paths[i][0]);
            vector1D dir_j = get_unit_vector(paths[j][N_path_j - 1] - paths[j][0]);
            if (vector_dot(dir_i, dir_j) < -0.5) {
                result[i][j] = -1;
            }
        }
    }

    return result;
}

double get_path_distance(const vector2D& path1, const vector2D& path2) {
    // double min_distance = 1e5;
    // for (const auto& point1 : path1) {
    //     for (const auto& point2 : path2) {
    //         double distance = get_norm(point1 - point2);
    //         min_distance = min(min_distance, distance);
    //     }
    // }
    // return min_distance;
    
    return get_DTW(path1, path2);
}

double get_SDTW(const vector2D& path1, const vector2D& path2) {
    int N_path1 = (int)path1.size();
    int N_path2 = (int)path2.size();

    vector2D ca = get_zero2D(N_path1, N_path2);
    vector2D n_ca = get_zero2D(N_path1, N_path2);
    
    // Subsequential DTW (path1 -> path2)
    for (int i = 0; i < N_path1; ++i) {
        for (int j = 0; j < N_path2; ++j) {
            double dis = get_norm(path1[i] - path2[j]);
            if (i == 0) {
                ca[i][j] = dis;
                n_ca[i][j] = 1;
            } else if (j == 0) {
                ca[i][j] = dis + ca[i - 1][j];
                n_ca[i][j] = 1 + n_ca[i - 1][j];
            } else {
                ca[i][j] = dis + ca[i - 1][j - 1];
                n_ca[i][j] = 1 + n_ca[i - 1][j - 1];
                if (dis + ca[i - 1][j] < ca[i][j]) {
                    ca[i][j] = dis + ca[i - 1][j];
                    n_ca[i][j] = 1 + n_ca[i - 1][j];
                }
                if (dis + ca[i][j - 1] < ca[i][j]) {
                    ca[i][j] = dis + ca[i][j - 1];
                    n_ca[i][j] = 1 + n_ca[i][j - 1];
                }
            }
        }
    }
    double min_distance = 1e5;
    for (int j = 0; j < N_path2; ++j) {
        min_distance = min(min_distance, ca[N_path1 - 1][j] / n_ca[N_path1 - 1][j]);
    }
    
    // Subsequential DTW (path2 -> path1)
    ca = get_zero2D(N_path1, N_path2);
    n_ca = get_zero2D(N_path1, N_path2);
    for (int i = 0; i < N_path1; ++i) {
        for (int j = 0; j < N_path2; ++j) {
            double dis = get_norm(path1[i] - path2[j]);
            if (j == 0) {
                ca[i][j] = dis;
                n_ca[i][j] = 1;
            } else if (i == 0) {
                ca[i][j] = dis + ca[i][j - 1];
                n_ca[i][j] = 1 + n_ca[i][j - 1];
            } else {
                ca[i][j] = dis + ca[i - 1][j - 1];
                n_ca[i][j] = 1 + n_ca[i - 1][j - 1];
                if (dis + ca[i - 1][j] < ca[i][j]) {
                    ca[i][j] = dis + ca[i - 1][j];
                    n_ca[i][j] = 1 + n_ca[i - 1][j];
                }
                if (dis + ca[i][j - 1] < ca[i][j]) {
                    ca[i][j] = dis + ca[i][j - 1];
                    n_ca[i][j] = 1 + n_ca[i][j - 1];
                }
            }
        }
    }

    for (int i = 0; i < N_path1; ++i) {
        min_distance = min(min_distance, ca[i][N_path2 - 1] / n_ca[i][N_path2 - 1]);
    }

    return min_distance;
}

double get_DTW(const vector2D& path1, const vector2D& path2) {
    int N_path1 = (int)path1.size();
    int N_path2 = (int)path2.size();

    vector2D ca = get_zero2D(N_path1, N_path2);
    vector2D n_ca = get_zero2D(N_path1, N_path2);
    
    // Subsequential DTW (path1 -> path2)
    for (int i = 0; i < N_path1; ++i) {
        for (int j = 0; j < N_path2; ++j) {
            double dis = get_norm(path1[i] - path2[j]);
            if (i == 0 && j == 0) {
                ca[i][j] = dis;
                n_ca[i][j] = 1;
            } else if (i == 0) {
                ca[i][j] = dis + ca[i][j - 1];
                n_ca[i][j] = 1 + n_ca[i][j - 1];
            } else if (j == 0) {
                ca[i][j] = dis + ca[i - 1][j];
                n_ca[i][j] = 1 + n_ca[i - 1][j];
            } else {
                ca[i][j] = dis + ca[i - 1][j - 1];
                n_ca[i][j] = 1 + n_ca[i - 1][j - 1];
                if (dis + ca[i - 1][j] < ca[i][j]) {
                    ca[i][j] = dis + ca[i - 1][j];
                    n_ca[i][j] = 1 + n_ca[i - 1][j];
                }
                if (dis + ca[i][j - 1] < ca[i][j]) {
                    ca[i][j] = dis + ca[i][j - 1];
                    n_ca[i][j] = 1 + n_ca[i][j - 1];
                }
            }
        }
    }
    double min_distance = ca[N_path1 - 1][N_path2 - 1] / n_ca[N_path1 - 1][N_path2 - 1];

    return min_distance;
}

double get_NP_global(const vector3D& init_paths, const vector3D& bundled_paths, int k) {
    int N = (int)bundled_paths.size();
    vector1D NP_per_path(N);

    // Iterate over each path to find k-nearest neighbors in initial and bundled paths
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        vector1D dis_pre(N), dis_bundled(N);
        vector<int> kNN_pre, kNN_bundled;
        // get distance
        for (int j = 0; j < N; ++j) {
            dis_pre[j] = get_path_distance(init_paths[i], init_paths[j]);
            dis_bundled[j] = get_path_distance(bundled_paths[i], bundled_paths[j]);
            if (i != j) {
                kNN_pre.push_back(j);
                kNN_bundled.push_back(j);
            }
        }

        sort(kNN_pre.begin(), kNN_pre.end(), [&](int i, int j) { return dis_pre[i] < dis_pre[j]; });
        sort(kNN_bundled.begin(), kNN_bundled.end(), [&](int i, int j) { return dis_bundled[i] < dis_bundled[j]; });

        // debug kNN
        // cout << "kNN_pre: ";
        // for (int j = 0; j < N - 1; ++j) {
        //     cout << dis_pre[kNN_pre[j]] << " ";
        // }
        // cout << "\n";
        // cout << "kNN_bundled: ";
        // for (int j = 0; j < N - 1; ++j) {
        //     cout << dis_bundled[kNN_bundled[j]] << " ";
        // }
        // cout << "\n";

        if (kNN_pre.size() > k) {
            kNN_pre.erase(kNN_pre.begin() + k, kNN_pre.end());
        }
        if (kNN_bundled.size() > k) {
            kNN_bundled.erase(kNN_bundled.begin() + k, kNN_bundled.end());
        }

        int preserved_count = 0;
        for (int j : kNN_bundled) {
            if (find(kNN_pre.begin(), kNN_pre.end(), j)!= kNN_pre.end()) {
                preserved_count++;
            }
        }

        NP_per_path[i] = ((double)preserved_count) / ((double)kNN_pre.size());
    }

    double sum_ratio = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_ratio += NP_per_path[i];
    }

    return sum_ratio / N;
}

vector1D interpolate_path_by_distance_ratio(const vector2D& path,
                                            double ratio) {
    int s = (int)path.size();

    if (ratio < 0) {
        return path[0];
    } else if (ratio >= 1) {
        return path[s - 1];
    }

    double sum_dis = 0;
    for (int i = 0; i < s - 1; i++) {
        sum_dis += get_norm(path[i] - path[i + 1]);
    }

    double target_dis = sum_dis * ratio;

    for (int i = 0; i < s - 1; i++) {
        double cur_dis = get_norm(path[i] - path[i + 1]);

        if (target_dis <= cur_dis) {
            double t = target_dis / cur_dis;
            return path[i] + t * (path[i + 1] - path[i]);
        }

        target_dis -= cur_dis;
    }

    return path[s - 1];
}

double get_NP_local(const vector3D& init_paths, const vector3D& bundled_paths, const vector2D& passing_flags, int k, int bundle_legacy) {
    int N = (int)bundled_paths.size();
    
    vector2D all_point_pre;
    vector2D all_point_bundled;

    for (int i = 0; i < N; ++i) {
        int N_path = (int)init_paths[i].size();
        for (int j = 0; j < N_path; ++j) {
            all_point_pre.push_back(init_paths[i][j]);
        }
    }

    // if (bundle_legacy) {
    //     // baseline don't preserve passing points, so we need to interpolate points
    //     for (int i = 0; i < N; ++i) {
    //         int N_path = (int)init_paths[i].size();
    //         // compute init path length
    //         double path_length = 0;
    //         for (int j = 0; j < N_path - 1; ++j) {
    //             path_length += get_norm(init_paths[i][j] - init_paths[i][j + 1]);
    //         }

    //         double cur_path_length = 0;

    //         for (int j = 0; j < N_path; ++j) {
    //             if (j > 0) {
    //                 cur_path_length += get_norm(init_paths[i][j] - init_paths[i][j - 1]);
    //             }

    //             double passing_point_ratio = cur_path_length / path_length;
    //             vector1D interpolated_passing_point = interpolate_path_by_distance_ratio(bundled_paths[i], passing_point_ratio);
    //             all_point_bundled.push_back(interpolated_passing_point);
    //         }
    //     }
    // } else {
    //     for (int i = 0; i < N; ++i) {
    //         int N_path = (int)bundled_paths[i].size();
    //         for (int j = 0; j < N_path; ++j) {
    //             if (passing_flags[i][j] > 0) {
    //                 all_point_bundled.push_back(bundled_paths[i][j]);
    //             }
    //         }
    //     }
    // }

    // interpolate points for all condition
    for (int i = 0; i < N; ++i) {
        int N_path = (int)init_paths[i].size();
        // compute init path length
        double path_length = 0;
        for (int j = 0; j < N_path - 1; ++j) {
            path_length += get_norm(init_paths[i][j] - init_paths[i][j + 1]);
        }

        double cur_path_length = 0;

        for (int j = 0; j < N_path; ++j) {
            if (j > 0) {
                cur_path_length += get_norm(init_paths[i][j] - init_paths[i][j - 1]);
            }

            double passing_point_ratio = cur_path_length / path_length;
            vector1D interpolated_passing_point = interpolate_path_by_distance_ratio(bundled_paths[i], passing_point_ratio);
            all_point_bundled.push_back(interpolated_passing_point);
        }
    }

    assert(all_point_pre.size() == all_point_bundled.size());
    int N_CP = (int)all_point_pre.size();

    vector1D NP_per_point(N_CP);

    // Iterate over each path to find k-nearest neighbors in initial and bundled paths
#pragma omp parallel for
    for (int i = 0; i < N_CP; ++i) {
        vector1D dis_pre(N_CP), dis_bundled(N_CP);
        vector<int> kNN_pre, kNN_bundled;
        // get distance
        for (int j = 0; j < N_CP; ++j) {
            dis_pre[j] = get_norm(all_point_pre[i] - all_point_pre[j]);
            dis_bundled[j] = get_norm(all_point_bundled[i] - all_point_bundled[j]);
            if (i != j) {
                kNN_pre.push_back(j);
                kNN_bundled.push_back(j);
            }
        }

        sort(kNN_pre.begin(), kNN_pre.end(), [&](int i, int j) { return dis_pre[i] < dis_pre[j]; });
        sort(kNN_bundled.begin(), kNN_bundled.end(), [&](int i, int j) { return dis_bundled[i] < dis_bundled[j]; });

        if (kNN_pre.size() > k) {
            kNN_pre.erase(kNN_pre.begin() + k, kNN_pre.end());
        }
        if (kNN_bundled.size() > k) {
            kNN_bundled.erase(kNN_bundled.begin() + k, kNN_bundled.end());
        }

        int preserved_count = 0;
        for (int j : kNN_bundled) {
            if (find(kNN_pre.begin(), kNN_pre.end(), j)!= kNN_pre.end()) {
                preserved_count++;
            }
        }

        NP_per_point[i] = ((double)preserved_count) / ((double)kNN_pre.size());
    }

    double sum_ratio = 0.0;
    for (int i = 0; i < N_CP; ++i) {
        sum_ratio += NP_per_point[i];
    }

    return sum_ratio / N_CP;
}

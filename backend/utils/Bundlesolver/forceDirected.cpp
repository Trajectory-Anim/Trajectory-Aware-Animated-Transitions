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

#include "forceDirected.h"
#include "ufs.h"
#include "vectorND.h"
#include "NPmetric.h"

using namespace std;
namespace py = pybind11;

struct ForceControlPoint {
    vector1D pos;
    bool isStartOrEnd;
    vector1D originPos;

    vector1D dir_prev;
    vector1D dir_next;

    vector1D force_spring;
    vector1D force_anchor;
    vector1D force_attract;
    vector1D force_all;

    int cluster_id;
    int belong_id;
    int idU;

    // vector<tuple<int, int>> att_neighbors;
};

double global_merge_time = 0.0;
double global_spring_force_time = 0.0;
double global_attract_force_time = 0.0;
double global_anchor_force_time = 0.0;
double global_remove_overlap_time = 0.0;
double global_remove_loop_time = 0.0;
double global_kNN_time = 0.0;
double global_dir_time = 0.0;

vector3D extract_control_point_pos(
    const vector<vector<shared_ptr<ForceControlPoint>>>& control_points) {
    vector3D result;
    for (const auto& control_points_list : control_points) {
        vector2D tmp_result;
        for (const auto& control_point : control_points_list) {
            tmp_result.push_back(control_point->pos);
        }
        result.push_back(tmp_result);
    }
    return result;
}

vector1D get_spring_force(const shared_ptr<ForceControlPoint>& cur,
                          const shared_ptr<ForceControlPoint>& prev,
                          const shared_ptr<ForceControlPoint>& next,
                          double k_s,
                          double multi_rate) {
    int cluster_id = cur->cluster_id;
    int cluster_id_bwd = prev->cluster_id;
    int cluster_id_fwd = next->cluster_id;

    vector1D force({0, 0});

    if (cluster_id != cluster_id_bwd) {
        // vector1D force_bwd = k_s * (prev->pos - cur->pos);
        // force += force_bwd;
        vector1D force_bwd_dir = get_unit_vector(prev->pos - cur->pos);
        double dis_bwd = get_norm(prev->pos - cur->pos);
        force += force_bwd_dir * dis_bwd * k_s;
    }

    if (cluster_id != cluster_id_fwd) {
        // vector1D force_fwd = k_s * (next->pos - cur->pos);
        // force += force_fwd;
        vector1D force_fwd_dir = get_unit_vector(next->pos - cur->pos);
        double dis_fwd = get_norm(next->pos - cur->pos);
        double dis_fwd_origin = get_norm(next->originPos - cur->originPos);
        force += force_fwd_dir * dis_fwd * k_s;
    }

    return force * multi_rate;
}

vector1D get_anchor_force(const shared_ptr<ForceControlPoint>& cur, double k_a) {
    vector1D tmp = cur->originPos - cur->pos;
    vector1D force_dir = get_unit_vector(tmp);
    double dis = get_norm(tmp);

    double force_mag = dis * dis * k_a;

    vector1D force = force_dir * force_mag;
    return force;
}

double lorenzian(double x, double s) {
    double tmp = s * s + x * x;
    return s * x / (3.1415 * tmp * tmp);
}

// the force is on I
vector1D get_attract_force(const shared_ptr<ForceControlPoint>& curI,
                             const shared_ptr<ForceControlPoint>& curJ,
                             double k_c,
                             double s,
                             double check_dot,
                             double connectivity_strength,
                             double compatibility,
                             double divide_rate) {
    double dir_dot = max(vector_dot(curI->dir_prev, curJ->dir_prev), vector_dot(curI->dir_next, curJ->dir_next));
    bool inverse = (dir_dot < 0);
    if (dir_dot < check_dot && dir_dot > -check_dot) {
        return vector1D({0, 0});
    }

    if (curI->cluster_id == curJ->cluster_id) {
        return vector1D({0, 0});
    }

    vector1D m_j = curJ->pos;

    if (inverse) {
        return vector1D({0, 0});
    }

    // use point dis
    vector1D force_dir = get_unit_vector(m_j - curI->pos);
    double dis = get_norm(m_j - curI->pos);

    // compute connectivity according to the angle
    double connectivity = abs(dir_dot);
    connectivity = pow(connectivity, connectivity_strength);

    double force_mag = 0;
    // prevent divide by zero
    if (dis > 1e-5) {
        force_mag = lorenzian(dis, s) * k_c * connectivity * compatibility / divide_rate;
    }

    vector1D force = force_dir * force_mag;
    return force;
}

int get_number_of_control_points(const vector<vector<shared_ptr<ForceControlPoint>>>& control_points) {
    int result = 0;
    for (const auto& control_points_list : control_points) {
        result += (int)control_points_list.size();
    }
    return result;
}

void get_control_points_dir(vector<vector<shared_ptr<ForceControlPoint>>>& control_points,
                            double eps) {
    auto time_point_start = chrono::steady_clock::now(); 

    int N = (int)control_points.size();

#pragma omp parallel for
    for (int p = 0; p < N; p++) {
        int N_CP = (int)control_points[p].size();

        for (int i = 1; i < N_CP - 1; i++) {
            // find prev point and next point
            int prev_i = i - 1;
            int next_i = i + 1;

            while (prev_i > 0 && get_norm(control_points[p][prev_i]->pos -
                                          control_points[p][i]->pos) < eps) {
                prev_i--;
            }

            while (next_i < N_CP - 1 &&
                   get_norm(control_points[p][next_i]->pos -
                            control_points[p][i]->pos) < eps) {
                next_i++;
            }

            // get direction
            vector1D prev_pos = control_points[p][prev_i]->pos;
            vector1D next_pos = control_points[p][next_i]->pos;
            vector1D cur_pos = control_points[p][i]->pos;
            control_points[p][i]->dir_prev = get_unit_vector(cur_pos - prev_pos);
            control_points[p][i]->dir_next = get_unit_vector(next_pos - cur_pos);
        }
    }

    auto time_point_end = chrono::steady_clock::now();
    global_dir_time +=
        (chrono::duration_cast<chrono::duration<double>>(time_point_end - time_point_start)).count();
}

vector<vector<shared_ptr<ForceControlPoint>>> init_control_points(const vector3D& paths, double d) {
    int N = (int)paths.size();

    vector3D control_points_pos;
    for (int i = 0; i < N; i++) {
        vector2D control_point;

        // init control point by path
        vector2D path = paths[i];
        int N_path = (int)path.size();


            control_point.push_back(path[0]);


            for (int j = 0; j < N_path - 1; j++) {
                double dis = get_norm(path[j + 1] - path[j]);
                int seg_num = (int)(dis / d);
                if (seg_num >= 2) {
                    vector1D delta_pos = (path[j + 1] - path[j]) / (double)seg_num;
                    for (int k = 1; k < seg_num; k++) {
                        control_point.push_back(path[j] + delta_pos * k);
                    }
                }
                control_point.push_back(path[j + 1]);
            }


        control_points_pos.push_back(control_point);
    }

    int count = 0;

    // init control points according to the position
    vector<vector<shared_ptr<ForceControlPoint>>> control_points;
    for (int i = 0; i < N; i++) {
        int N_CP = (int)control_points_pos[i].size();
        vector<shared_ptr<ForceControlPoint>> control_point;

        for (int j = 0; j < N_CP; j++) {
            shared_ptr<ForceControlPoint> new_control_point(new ForceControlPoint());

            new_control_point->pos = control_points_pos[i][j];
            new_control_point->isStartOrEnd = (j == 0 || j == N_CP - 1);
            new_control_point->originPos = control_points_pos[i][j];
            new_control_point->belong_id = i;
            new_control_point->idU = count;
            new_control_point->cluster_id = count;
            count++;

            control_point.push_back(new_control_point);
        }
        control_points.push_back(control_point);
    }

    return control_points;
}

void merge_control_points(vector<vector<shared_ptr<ForceControlPoint>>>& control_points,
                          const vector<vector<int>>& path_kNN,
                          const vector2D& path_compatibility,
                          double merge_dis,
                          double check_dot,
                          int N_all_CP,
                          int stage) {
    auto time_point_start = chrono::steady_clock::now();

    int N = (int)control_points.size();

    // init union find set
    UFS ufs(N_all_CP);
    for (int p = 0; p < N; p++) {
        int N_CP = (int)control_points[p].size();
        for (int i = 0; i < N_CP; i++) {
            auto& cur = control_points[p][i];
            ufs.union_set(cur->idU, cur->cluster_id);
        }
    }

    vector<vector<bool>> merged;
    for (int p = 0; p < N; p++) {
        int N_CP = (int)control_points[p].size();
        merged.push_back(vector<bool>(N_CP, false));
    }

    // try to merge
    for (int p = 0; p < N; p++) {
        int N_CP = (int)control_points[p].size();
        for (int i = 1; i < N_CP - 1; i++) {
            if (merged[p][i]) {
                continue;
            }

            merged[p][i] = true;

            auto& curI = control_points[p][i];
            vector<int> path_to_check_merge = path_kNN[p];
            path_to_check_merge.push_back(p);

            if (stage > 1) {
                path_to_check_merge.clear();
                for (int j = 0; j < N; j++) {
                    if (path_compatibility[p][j] > 1e-5) {
                        path_to_check_merge.push_back(j);
                    }
                }
            }
            
            for (int q : path_to_check_merge) {
            // for (int q = 0; q < N; q++) {
                int N_CP_q = (int)control_points[q].size();
                if (path_compatibility[p][q] < 1e-5) {
                    continue;
                }

                for (int j = 1; j < N_CP_q - 1; j++) {
                    auto& curJ = control_points[q][j];
                    if (merged[q][j]) {
                        continue;
                    }

                    // ignore the start and end
                    if (curJ->isStartOrEnd) {
                        continue;
                    }

                    if (p == q && i == j) {
                        continue;
                    }

                    if (ufs.connected(curI->idU, curJ->idU)) {
                        // curJ->pos = curI->pos;
                        merged[q][j] = true;
                        continue;
                    }

                    double dis = get_norm(curI->pos - curJ->pos);
                    double dir_dot = max(vector_dot(curI->dir_prev, curJ->dir_prev), vector_dot(curI->dir_next, curJ->dir_next));

                    if (dis < merge_dis && dir_dot > check_dot) {
                        ufs.union_set(curI->idU, curJ->idU);
                        // curJ->pos = curI->pos;
                        merged[q][j] = true;
                    }
                }
            }
        }
    }

    // use the result from UFS to update the cluster
    for (int p = 0; p < N; p++) {
        int N_CP = (int)control_points[p].size();
        for (int i = 1; i < N_CP - 1; i++) {
            auto& cur = control_points[p][i];
            cur->cluster_id = ufs.find(cur->idU);
        }
    }

    vector2D cluster_pos = get_zero2D(N_all_CP, 2);
    vector<int> cluster_size(N_all_CP, 0);

    for (int p = 0; p < N; p++) {
        int N_CP = (int)control_points[p].size();
        for (int i = 1; i < N_CP - 1; i++) {
            auto& cur = control_points[p][i];
            cluster_pos[cur->cluster_id] += cur->pos;
            cluster_size[cur->cluster_id]++;
        }
    }

#pragma omp parallel for
    for (int p = 0; p < N; p++) {
        int N_CP = (int)control_points[p].size();
        for (int i = 1; i < N_CP - 1; i++) {
            auto& cur = control_points[p][i];
            cur->pos = cluster_pos[cur->cluster_id] / (double)cluster_size[cur->cluster_id];
        }
    }

    auto time_point_end = chrono::steady_clock::now();
    global_merge_time += (chrono::duration_cast<chrono::duration<double>>(time_point_end - time_point_start)).count();
}

void remove_overlap_control_points(
    vector<vector<shared_ptr<ForceControlPoint>>>& control_points) {
    auto time_point_start = chrono::steady_clock::now();

    int N_real_CP_prev = get_number_of_control_points(control_points);

    int N = (int)control_points.size();

#pragma omp parallel for
    for (int p = 0; p < N; p++) {
        int N_CP = (int)control_points[p].size();
        vector<shared_ptr<ForceControlPoint>> new_control_points;
        new_control_points.push_back(control_points[p][0]);
        int N_CP_new = 1;

        for (int i = 1; i < N_CP - 1; i++) {
            auto& cur = new_control_points[N_CP_new - 1];
            auto& next = control_points[p][i];

            if (get_norm(cur->pos - next->pos) < 1e-5) {
                // do nothing

            } else {
                new_control_points.push_back(next);
                N_CP_new++;
            }
        }

        new_control_points.push_back(control_points[p][N_CP - 1]);
        control_points[p] = new_control_points;
    }

    int N_real_CP = get_number_of_control_points(control_points);
    // cout << "Removed. Number of control points: " << N_real_CP_prev << "->" << N_real_CP << endl;

    auto time_point_end = chrono::steady_clock::now();
    global_remove_overlap_time += (chrono::duration_cast<chrono::duration<double>>(time_point_end - time_point_start)).count();
}

void remove_loop_control_points(vector<vector<shared_ptr<ForceControlPoint>>>& control_points) {
    // get running time
    auto time_point_start = chrono::steady_clock::now();

    int N = (int)control_points.size();

// #pragma omp parallel for
    for (int p = 0; p < N; p++) {
        int N_CP = (int)control_points[p].size();
        vector<shared_ptr<ForceControlPoint>> new_control_points;
        new_control_points.push_back(control_points[p][0]);
        for (int i = 1; i < N_CP - 1; i++) {
            bool flag = false;
            int next_i = -1;

            auto& cur = control_points[p][i];
            new_control_points.push_back(cur);
            for (int j = N_CP - 2; j > i; j--) {
                auto& other = control_points[p][j];
                if (get_norm(cur->pos - other->pos) < 1e-5) {
                    flag = true;
                    next_i = j + 1;
                    break;
                }
            }

            if (flag) {
                // cout << "Detected a loop in path " << p << endl;
                // cout << "    i = " << i << ", j = " << next_i - 1 << ")" << endl;
                i = next_i - 1;
            }
        }
        new_control_points.push_back(control_points[p][N_CP - 1]);
        control_points[p] = new_control_points;
    }

    auto time_point_end = chrono::steady_clock::now();
    global_remove_loop_time += (chrono::duration_cast<chrono::duration<double>>(time_point_end - time_point_start)).count();
}

// return the control points list
vector4D get_control_points_lists(const py::object& anim, vector3D paths, int stage) {
    // get running time
    auto all_start_time = chrono::steady_clock::now();

    cout << "Cur Bundling Stage: " << stage << "/" << anim.attr("n_stage").cast<int>() << endl;

    // clear all global time
    global_merge_time = 0.0;
    global_spring_force_time = 0.0;
    global_attract_force_time = 0.0;
    global_anchor_force_time = 0.0;
    global_remove_overlap_time = 0.0;
    global_kNN_time = 0.0;
    global_dir_time = 0.0;
    global_remove_loop_time= 0.0;


    int N = (int)paths.size();

    double d = anim.attr("d").cast<double>();
    bool debug_mode = anim.attr("debug_mode").cast<bool>();

    vector2D path_compatibility = get_zero2D(N, N);

    double subgroup_dis = anim.attr("subgroup_dis").cast<double>();
    double max_DTW = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                continue;
            } else {
                max_DTW = max(max_DTW, get_path_distance(paths[i], paths[j]));
            } 
        }
    }

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                path_compatibility[i][j] = 1;
            } else {
                path_compatibility[i][j]  = max(0.0, 1.0 - get_path_distance(paths[i], paths[j]) / max_DTW);
            } 
        }
    }

    // 根据路径兼容性排序路径
    auto time_point_kNN_begin = chrono::steady_clock::now();
    int kNN_k = anim.attr("kNN_k").cast<int>();
    vector<vector<int>> path_kNN(N);
    for (int i = 0; i < N; i++) {
        vector<int> other_path_idx;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                other_path_idx.push_back(j);
            }
        }
        sort(other_path_idx.begin(), other_path_idx.end(), [&](int a, int b) {
            return path_compatibility[i][a] > path_compatibility[i][b];
        });
        int real_k = min(kNN_k, (int)other_path_idx.size());
        path_kNN[i] = vector<int>(other_path_idx.begin(), other_path_idx.begin() + real_k);
    }
    auto time_point_kNN_end = chrono::steady_clock::now();
    global_kNN_time += (chrono::duration_cast<chrono::duration<double>>(time_point_kNN_end - time_point_kNN_begin)).count();


    // generate initial control points
    auto control_points = init_control_points(paths, d);
    vector4D control_points_lists({extract_control_point_pos(control_points)});

    int N_all_CP = 0;
    for (auto& cps : control_points) {
        N_all_CP += (int)cps.size();
    }

    // loop for each cycle
    double cur_dt = anim.attr("dt").cast<double>();
    // double cur_s = anim.attr("s").cast<double>();
    double cur_k_c = anim.attr("k_c").cast<double>();
    double cur_k_s = anim.attr("k_s").cast<double>();
    double cur_k_a = anim.attr("k_a").cast<double>();

    vector1D s_list = anim.attr("s_list").cast<vector1D>();
    double cur_s = s_list[stage - 1];

    vector1D k_c_rate_list = anim.attr("k_c_rate_list").cast<vector1D>();
    cur_k_c *= k_c_rate_list[stage - 1];

    // double md = anim.attr("md").cast<double>();
    vector1D md_list = anim.attr("md_list").cast<vector1D>();
    double md = md_list[stage - 1];

    double connectivity_strength =
        anim.attr("connectivity_strength").cast<double>();
    double connectivity_angle = anim.attr("connectivity_angle").cast<double>();
    double check_dot = sin(connectivity_angle / 2 / 180 * 3.1415);

    int n_iter = anim.attr("n_iter").cast<int>();

    double move_lim = anim.attr("move_lim").cast<double>();
    move_lim /= (double) n_iter;

    int merge_iter = anim.attr("merge_iter").cast<int>();

    // add force and iterate
    double debug_spring_force = 0;
    double debug_anchor_force = 0;
    double debug_attract_force = 0;
    double debug_total_force = 0;


    for (int iter = 0; iter < n_iter; iter++) {
        // we calculate direction
        get_control_points_dir(control_points, md);

        if ((iter + 1) % merge_iter == 0 || iter == 0) {
            merge_control_points(control_points, path_kNN, path_compatibility, md, check_dot, N_all_CP, stage);
        }

        // clean up force on the control points
        int N_real_CP = 0;
        for (int p = 0; p < N; p++) {
            int N_CP = (int)control_points[p].size();
            N_real_CP += N_CP;
            for (int i = 1; i < N_CP - 1; i++) {
                control_points[p][i]->force_all = vector1D({0, 0});
                control_points[p][i]->force_spring = vector1D({0, 0});
                control_points[p][i]->force_anchor = vector1D({0, 0});
                control_points[p][i]->force_attract = vector1D({0, 0});
            }
        }

        auto time_point_spring_begin = chrono::steady_clock::now();
        // add spring force
#pragma omp parallel for
        for (int p = 0; p < N; p++) {
            int N_CP = (int)control_points[p].size();
            // double multi_rate = (bundle_legacy ? N_CP : 1);
            double multi_rate = N_CP;

            for (int i = 1; i < N_CP - 1; i++) {
                auto& cur = control_points[p][i];
                auto& prev = control_points[p][i - 1];
                auto& next = control_points[p][i + 1];
                cur->force_spring =
                    get_spring_force(cur, prev, next, cur_k_s, multi_rate);
            }
        }
        auto time_point_spring_end = chrono::steady_clock::now();

        auto time_point_anchor_begin = chrono::steady_clock::now();
        // add anchor force
#pragma omp parallel for
        for (int p = 0; p < N; p++) {
            int N_CP = (int)control_points[p].size();
            for (int i = 1; i <= N_CP - 1; i++) {
                auto& cur = control_points[p][i];
                cur->force_anchor = get_anchor_force(cur, cur_k_a);
            }
        }
        auto time_point_anchor_end = chrono::steady_clock::now();

        auto time_point_attract_begin = chrono::steady_clock::now();
        // add attract force
#pragma omp parallel for
        for (int p = 0; p < N; p++) {
            int N_CPP = (int)control_points[p].size();
            for (int i = 1; i < N_CPP - 1; i++) {
                auto& curI = control_points[p][i];
                for (auto q : path_kNN[p]) {
                    int N_CPQ = (int)control_points[q].size();
                    for (int j = 1; j < N_CPQ - 1; j++) {
                        if (p == q) {
                            continue;
                        }

                        auto& curJ = control_points[q][j];

                        // ignore the start and end point
                        if (curJ->isStartOrEnd) {
                            continue;
                        }

                        if (curI->cluster_id == curJ->cluster_id) {
                            continue;
                        }

                        double divide_rate = N_CPQ;
                        double compatibility = path_compatibility[p][q];

                        if (compatibility < 1e-5) {
                            continue;
                        }

                        vector1D cur_force = get_attract_force(
                            curI, curJ, cur_k_c, cur_s, check_dot,
                            connectivity_strength, compatibility, divide_rate);
                        

                        curI->force_attract += cur_force;
                    }
                }
            }
        }

        auto time_point_attract_end = chrono::steady_clock::now();

        // add the force on cluster
        vector2D cur_force_cluster = get_zero2D(N_all_CP, 2);
        for (int p = 0; p < N; p++) {
            int N_CP = (int)control_points[p].size();

            for (int i = 1; i < N_CP - 1; i++) {
                auto& cur = control_points[p][i];
                cur->force_all = cur->force_spring + cur->force_anchor +
                                cur->force_attract;
                cur_force_cluster[cur->cluster_id] += cur->force_all;

                debug_spring_force += get_norm(cur->force_spring) / (double)N_real_CP;
                debug_anchor_force += get_norm(cur->force_anchor) / (double)N_real_CP;
                debug_attract_force += get_norm(cur->force_attract) / (double)N_real_CP;
                debug_total_force += get_norm(cur->force_all) / (double)N_real_CP;
            }
        }

        // count time

        global_anchor_force_time += chrono::duration_cast<chrono::duration<double>>(time_point_anchor_end - time_point_anchor_begin).count();
        global_spring_force_time += chrono::duration_cast<chrono::duration<double>>(time_point_spring_end - time_point_spring_begin).count();
        global_attract_force_time += chrono::duration_cast<chrono::duration<double>>(time_point_attract_end - time_point_attract_begin).count();
        
        // compute cluster size
        vector<int> cluster_size(N_all_CP, 0);
        for (int p = 0; p < N; p++) {
            int N_CP = (int)control_points[p].size();
            for (int i = 0; i < N_CP; i++) {
                cluster_size[control_points[p][i]->cluster_id]++;
            }
        }

        // average the force
#pragma omp parallel for
        for (int cluster_id = 0; cluster_id < N_all_CP; cluster_id++) {
            cur_force_cluster[cluster_id] /= cluster_size[cluster_id];

            double force_mag = get_norm(cur_force_cluster[cluster_id]);
            if (force_mag * cur_dt > move_lim) {
                cur_force_cluster[cluster_id] *=
                    move_lim / cur_dt / force_mag;
            }
        }

        // update location
#pragma omp parallel for
        for (int p = 0; p < N; p++) {
            int N_CP = (int)control_points[p].size();
            for (int i = 1; i < N_CP - 1; i++) {
                auto& cur = control_points[p][i];
                int cluster_id = cur->cluster_id;
                cur->pos += cur_force_cluster[cluster_id] * cur_dt;
            }
        }

        // append the current control points to the whole list
        if (debug_mode) {
            control_points_lists.push_back(
                extract_control_point_pos(control_points));
        }

    }

    // debug force
    debug_spring_force /= n_iter;
    debug_anchor_force /= n_iter;
    debug_attract_force /= n_iter;
    debug_total_force /= n_iter;

    // cout << "  spring:    " << debug_spring_force << "\n";
    // cout << "  anchor:    " << debug_anchor_force << "\n";
    // cout << "  attract: " << debug_attract_force << "\n";
    // cout << "  total:     " << debug_total_force << "\n";
    // cout << "  attract_active: " << attract_force_active_count / (double)attract_force_count << "\n";
    merge_control_points(control_points, path_kNN, path_compatibility, md, check_dot, N_all_CP, stage);
    remove_overlap_control_points(control_points);
    remove_loop_control_points(control_points);

    auto all_end_time = chrono::steady_clock::now();
    double total_time = chrono::duration_cast<chrono::duration<double>>(all_end_time - all_start_time).count();

    // cout << "Get direction time: " << global_dir_time << "s\n";
    // cout << "Get merge time: " << global_merge_time << "s\n";
    // cout << "Remove overlap time: " << global_remove_overlap_time << "s\n";
    // cout << "Get spring time: " << global_spring_force_time << "s\n";
    // cout << "Get anchor time: " << global_anchor_force_time << "s\n";
    // cout << "Get attract time: " << global_attract_force_time << "s\n";
    // cout << "Get kNN time: " << global_kNN_time << "s\n";
    // cout << "Remove loop time: " << global_remove_loop_time << "s\n";
    // cout << "Total time: " << total_time << "s\n";

    control_points_lists.push_back(
        extract_control_point_pos(control_points));

    // cout << "Number of control points: " << get_number_of_control_points(control_points) << "\n";
    return control_points_lists;
}

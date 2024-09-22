#include <omp.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <memory>
#include <queue>
#include <set>
#include <tuple>
#include <utility>
#include <vector>
#include "../tinynurbs/tinynurbs.h"
#include <chrono>
// #include <opencv2/opencv.hpp>

#include "anim_compute.h"
#include "ufs.h"
#include "vectorND.h"
#include "circle_b2d.h"
#define M_PI 3.14159265358979323846

using namespace std;
namespace py = pybind11;

int global_control_point_num = 0;
int global_key_point_num = 0;

struct KeyPoint;

struct ControlPoint {
    vector1D pos;
    shared_ptr<KeyPoint> keypoint;
    int idP, idC;  // the control point is the 'idC' control point of the 'idP'
                   // object
    int idU;       // the control point is the 'idU' control point of all
    ControlPoint(vector1D pos1, int id1, int id2)
        : pos(pos1),
          keypoint(nullptr),
          idP(id1),
          idC(id2),
          idU(global_control_point_num++) {}
};

vector<vector<int>> cluster_to_sub_cluster(const vector<int> &cluster, const vector<vector<int>> &all_sub_cluster) {
    vector<vector<int>> sub_cluster_vector;

    for (auto sub_cluster : all_sub_cluster) {
        bool cur_have = false;
        bool cur_all_have = true;
        vector<int> sub_cluster_have = vector<int>();
        for (auto p : sub_cluster) {
            if (find(cluster.begin(), cluster.end(), p) == cluster.end()) {
                cur_all_have = false;
            } else {
                cur_have = true;
                sub_cluster_have.push_back(p);
            }
        }
        if (cur_have) {
            if (cur_all_have) {
                sub_cluster_vector.push_back(sub_cluster);
            } else {
                sub_cluster_vector.push_back(sub_cluster_have);
                // cout << "Warning! Some sub-cluster is not completely in the current cluster!" << endl;
                // sub_cluster_vector.push_back(sub_cluster);

            }
        }
    }
    return sub_cluster_vector;
}

struct KeyPoint {
    vector1D pos;

    vector<shared_ptr<ControlPoint>> points;
    vector<shared_ptr<KeyPoint>> DAG_edges;
    vector<shared_ptr<KeyPoint>> DAG_inv_edges;

    int id;  // for debug

    KeyPoint(vector<shared_ptr<ControlPoint>> points1)
        : points(points1),
          id(global_key_point_num++) {
        pos = {0, 0};
        for (auto p : points) {
            pos = pos + p->pos;
        }
        int psize = (int)points.size();
        pos = pos / psize;
    }

    set<int> getCluster() {
        set<int> pointID;
        for (auto p : points) {
            pointID.insert(p->idP);
        }
        return pointID;
    }

    vector<int> getClusterVector() {
        vector<int> pointID;
        for (auto p : points) {
            if (find(pointID.begin(), pointID.end(), p->idP) == pointID.end()) {
                pointID.push_back(p->idP);
            }
        }
        sort(pointID.begin(), pointID.end());
        return pointID;
    }

    vector<vector<int>> getSubClusterVector(const vector<vector<int>>& all_sub_cluster) {
        return cluster_to_sub_cluster(getClusterVector(), all_sub_cluster);
    }

    void debug() {
        auto pointID = getCluster();
        cout << "[ ";
        for (auto p : pointID) {
            cout << p << " ";
        }
        cout << "]" << endl;
    }
};

double get_vector2D_length(const vector2D& vec) {
    double length = 0;
    int N_path = (int)vec.size();
    for (int i = 0; i < N_path - 1; i++) {
        length += get_norm(vec[i] - vec[i + 1]);
    }
    return length;
}

vector2D get_vector2D_cut(const vector2D& vec, double head_cut, double tail_cut) {
    double sum_length = get_vector2D_length(vec);
    head_cut = min(head_cut, sum_length - 1e-5);
    tail_cut = min(tail_cut, sum_length - 1e-5);

    vector2D cut_result;

    int N_path = (int)vec.size();

    int head_cut_id = 0;
    int tail_cut_id = N_path - 1;
    vector1D head_cut_point;
    vector1D tail_cut_point;

    for (int i = 0; i < N_path - 1; i++) {
        double cur_length = get_norm(vec[i] - vec[i + 1]);
        if (cur_length > head_cut) {
            head_cut_id = i;
            double t = head_cut / cur_length;
            head_cut_point = vec[i] + t * (vec[i + 1] - vec[i]);
            break;
        } else {
            head_cut -= cur_length;
        }
    }

    for (int i = N_path - 1; i > 0; i--) {
        double cur_length = get_norm(vec[i] - vec[i - 1]);
        if (cur_length > tail_cut) {
            tail_cut_id = i;
            double t = tail_cut / cur_length;
            tail_cut_point = vec[i] + t * (vec[i - 1] - vec[i]);
            break;
        } else {
            tail_cut -= cur_length;
        }
    }

    cut_result.push_back(head_cut_point);
    for (int i = head_cut_id + 1; i < tail_cut_id; i++) {
        cut_result.push_back(vec[i]);
    }
    cut_result.push_back(tail_cut_point);

    return cut_result;
}

vector2D get_vector2D_reverse(const vector2D& vec) {
    int N_path = (int)vec.size();
    vector2D reverse_result;
    for (int i = N_path - 1; i >= 0; i--) {
        reverse_result.push_back(vec[i]);
    }
    return reverse_result;
}

vector2D get_vector2D_half(const vector2D& vec, bool from_start) {
    double length = get_vector2D_length(vec);
    if (!from_start) {
        return get_vector2D_cut(vec, length / 2, 0);
    } else {
        return get_vector2D_cut(vec, 0, length / 2);
    }
}
vector2D get_vector2D_triplet(const vector2D& vec, bool from_start) {
    double length = get_vector2D_length(vec);
    if (!from_start) {
        return get_vector2D_cut(vec, length / 3, 0);
    } else {
        return get_vector2D_cut(vec, 0, 2*length / 3);
    }
}

vector1D get_optimal_require_dis(vector3D curves, const vector1D& radii, bool from_start) {
    int N_curves = (int)curves.size();

    if (!from_start) {
        for (int i = 0; i < N_curves; i++) {
            curves[i] = get_vector2D_reverse(curves[i]);
        }
    }

    auto get_loc = [&] (const vector2D& curve, double cur_dis) {
        vector2D curve_cut = get_vector2D_cut(curve, cur_dis, 0);
        return curve_cut[0];
    };

    vector1D curves_length(N_curves);
    for (int i = 0; i < N_curves; i++) {
        curves_length[i] = get_vector2D_length(curves[i]);
    }

    double move_step = 1e-4;
    
    vector1D require_dis = radii * 2;

    for (int i = 0; i < N_curves; i++) {
        require_dis[i] = min(require_dis[i], curves_length[i] - move_step);
    }

    vector2D cur_locs;

    auto check_overlap = [&] (int i) {
        for (int j = 0; j < N_curves; j++) {
            if (i == j) {
                continue;
            }
            if (get_norm(cur_locs[i] - cur_locs[j]) < radii[i] + radii[j]) {
                return true;
            }
        }
        return false;
    };

    auto get_max_overlap = [&] (int i) {
        double overlap = 0;
         for (int j = 0; j < N_curves; j++) {
            if (i == j) {
                continue;
            }

            double dis = get_norm(cur_locs[i] - cur_locs[j]);
            if (dis < radii[i] + radii[j]) {
                overlap = max(overlap, radii[i] + radii[j] - dis);
            }
        }
        return overlap;
    };

    for (int i = 0; i < N_curves; i++) {
        cur_locs.push_back(get_loc(curves[i], require_dis[i]));
    }

    // now, we try to optimize the require_dis
    for (int i = 0; i < N_curves; i++) {
        while (require_dis[i] < curves_length[i] && check_overlap(i)) {
            double real_move = max(move_step, get_max_overlap(i));

            require_dis[i] += real_move;
            if (require_dis[i] >= curves_length[i]) {
                require_dis[i] = curves_length[i];
                cur_locs[i] = get_loc(curves[i], require_dis[i]);
                break;
            }

            cur_locs[i] = get_loc(curves[i], require_dis[i]);
            // cout << "require dis / curvers_len : " << require_dis[i] << " / " << curves_length[i] << endl;
        }
    }

    return require_dis;
}

vector1D interpolate_vector2D(const vector2D& curve, double rate) {
    int N_path = (int)curve.size();

    if (rate <= 0) {
        return curve[0];
    } else if (rate >= 1) {
        return curve[N_path - 1];
    }

    double length = get_vector2D_length(curve);
    double inter_length = length * rate;

    for (int i = 0; i < N_path - 1; i++) {
        double cur_length = get_norm(curve[i] - curve[i + 1]);
        if (cur_length > inter_length) {
            double t = inter_length / cur_length;
            return curve[i] + t * (curve[i + 1] - curve[i]);
        } else {
            inter_length -= cur_length;
        }
    }

    return curve[N_path - 1];
}

vector2D merge_two_vector2D(const vector2D& vec1, const vector2D& vec2) {
    vector2D merged_curve;

    double step = 0.1;
    for (double t = 0; t <= 1; t += step) {
        vector1D inter_point1 = interpolate_vector2D(vec1, t);
        vector1D inter_point2 = interpolate_vector2D(vec2, t);

        merged_curve.push_back((inter_point1 + inter_point2) / 2);
    }

    return merged_curve;
}

vector2D apply_move_on_vector2D_head(const vector2D& vec, const vector1D& move) {
    int N_path = (int)vec.size();
    vector2D moved_curve = vec;

    double length = get_vector2D_length(vec);
    double cur_length = 0;

    for (int i = 0; i < N_path; i++) {
        double t = cur_length / length;

        moved_curve[i] += move * (1 - t);
        
        if (i != N_path - 1) {
            cur_length += get_norm(vec[i] - vec[i + 1]);
        }
    }

    return moved_curve;
}

vector2D apply_move_on_vector2D_tail(const vector2D& vec, const vector1D& move) {
    int N_path = (int)vec.size();
    vector2D moved_curve = vec;

    double length = get_vector2D_length(vec);
    double cur_length = 0;

    for (int i = N_path - 1; i >= 0; i--) {
        double t = cur_length / length;

        moved_curve[i] += move * (1 - t);
        
        if (i != 0) {
            cur_length += get_norm(vec[i] - vec[i - 1]);
        }
    }

    return moved_curve;
}

map<tuple<int, int>, vector<int>> get_edge_point_map(const vector<shared_ptr<KeyPoint>>& key_points_group) {
    map<tuple<int, int>, vector<int>> edge_point_map;
    map<int, int> rank;
    int cur_rank = 0;
    for (auto& key : key_points_group) {
        rank[key->id] = cur_rank++;
    }

    for (auto& key : key_points_group) {
        sort(key->DAG_edges.begin(), key->DAG_edges.end(),
             [&](shared_ptr<KeyPoint> a, shared_ptr<KeyPoint> b) {
                 return rank[a->id] < rank[b->id];
             });
        vector<int> cur_points = key->getClusterVector();

        // cout << "(" << key->id << ")" << ": ";
        // for (int i = 0; i < cur_points.size(); i++) {
        //     cout << cur_points[i] << " ";
        // }
        // cout << endl;

        for (auto& next_key : key->DAG_edges) {
            vector<int> next_points = next_key->getClusterVector();

            // get intersection of cur_points and next_points
            vector<int> intersect_points;
            vector<int> difference_points;
            set_intersection(
                cur_points.begin(), cur_points.end(), next_points.begin(),
                next_points.end(),
                inserter(intersect_points, intersect_points.begin()));
            set_difference(
                cur_points.begin(), cur_points.end(), next_points.begin(),
                next_points.end(),
                inserter(difference_points, difference_points.begin()));

            edge_point_map[make_tuple(key->id, next_key->id)] =
                intersect_points;
            cur_points = difference_points;

            // cout << "(" << key->id << "->" << next_key->id << ")" << ": ";
            // for (int i : intersect_points) {
            //     cout << i << " ";
            // }
            // cout << endl;
        }
    }

    return edge_point_map;
}

map<tuple<int, int>, vector<vector<int>>> get_edge_sub_cluster_map(const vector<shared_ptr<KeyPoint>>& key_points_group, const vector<vector<int>>& all_sub_clusters) {
    map<tuple<int, int>, vector<vector<int>>> edge_sub_cluster_map;
    map<int, int> rank;
    int cur_rank = 0;
    for (auto& key : key_points_group) {
        rank[key->id] = cur_rank++;
    }

    for (auto& key : key_points_group) {
        sort(key->DAG_edges.begin(), key->DAG_edges.end(),
             [&](shared_ptr<KeyPoint> a, shared_ptr<KeyPoint> b) {
                 return rank[a->id] < rank[b->id];
             });
        vector<int> cur_points = key->getClusterVector();

        // cout << "(" << key->id << ")" << ": ";
        // for (int i = 0; i < cur_points.size(); i++) {
        //     cout << cur_points[i] << " ";
        // }
        // cout << endl;

        for (auto& next_key : key->DAG_edges) {
            vector<int> next_points = next_key->getClusterVector();

            // get intersection of cur_points and next_points
            vector<int> intersect_points;
            vector<int> difference_points;
            set_intersection(
                cur_points.begin(), cur_points.end(), next_points.begin(),
                next_points.end(),
                inserter(intersect_points, intersect_points.begin()));
            set_difference(
                cur_points.begin(), cur_points.end(), next_points.begin(),
                next_points.end(),
                inserter(difference_points, difference_points.begin()));

            edge_sub_cluster_map[make_tuple(key->id, next_key->id)] = cluster_to_sub_cluster(intersect_points, all_sub_clusters);
            cur_points = difference_points;

            // cout << "(" << key->id << "->" << next_key->id << ")" << ": ";
            // for (int i : intersect_points) {
            //     cout << i << " ";
            // }
            // cout << endl;
        }
    }

    return edge_sub_cluster_map;
}

double get_single_key_point_entropy(shared_ptr<KeyPoint> key, map<tuple<int, int>, vector<int>>& edge_point_map) {
    double entropy = 0;
    int cur_num = (int)key->points.size();
    for (auto& prev_key : key->DAG_inv_edges) {
        auto edge_tuple = make_tuple(prev_key->id, key->id);
        int edge_num = (int)edge_point_map[edge_tuple].size();

        double percentage = (double)edge_num / (double)cur_num;
        entropy -= percentage * log2(percentage);
    }

    for (auto& next_key : key->DAG_edges) {
        auto edge_tuple = make_tuple(key->id, next_key->id);
        int edge_num = (int)edge_point_map[edge_tuple].size();

        double percentage = (double)edge_num / (double)cur_num;
        entropy -= percentage * log2(percentage);
    }

    return entropy;
}

vector<shared_ptr<KeyPoint>> get_prev_keys_of_two_key_points(shared_ptr<KeyPoint> key1, shared_ptr<KeyPoint> key2) {
    set<int> prev_ids = {key1->id, key2->id};
    vector<shared_ptr<KeyPoint>> prev_keys;
    for (auto& prev_key : key1->DAG_inv_edges) {
        if (prev_ids.count(prev_key->id) == 0) {
            prev_ids.insert(prev_key->id);
            prev_keys.push_back(prev_key);
        }
    }
    for (auto& prev_key : key2->DAG_inv_edges) {
        if (prev_ids.count(prev_key->id) == 0) {
            prev_ids.insert(prev_key->id);
            prev_keys.push_back(prev_key);
        }
    }

    return prev_keys;
}

vector<shared_ptr<KeyPoint>> get_next_keys_of_two_key_points(shared_ptr<KeyPoint> key1, shared_ptr<KeyPoint> key2) {
    set<int> next_ids = {key1->id, key2->id};
    vector<shared_ptr<KeyPoint>> next_keys;
    for (auto& next_key : key1->DAG_edges) {
        if (next_ids.count(next_key->id) == 0) {
            next_ids.insert(next_key->id);
            next_keys.push_back(next_key);
        }
    }
    for (auto& next_key : key2->DAG_edges) {
        if (next_ids.count(next_key->id) == 0) {
            next_ids.insert(next_key->id);
            next_keys.push_back(next_key);
        }
    }

    return next_keys;
}

double get_key_point_merge_size(shared_ptr<KeyPoint> key1, shared_ptr<KeyPoint> key2, map<tuple<int, int>, vector<int>>& edge_point_map) {
    int cur_num = (int)key1->points.size() + (int)key2->points.size();
    auto edge_tuple12 = make_tuple(key1->id, key2->id);
    auto edge_tuple21 = make_tuple(key2->id, key1->id);
    if (edge_point_map.count(edge_tuple12)) {
        cur_num -= (int)edge_point_map[edge_tuple12].size();
    }
    if (edge_point_map.count(edge_tuple21)) {
        cur_num -= (int)edge_point_map[edge_tuple21].size();
    }

    return cur_num;
}

double get_key_point_merge_entropy(shared_ptr<KeyPoint> key1, shared_ptr<KeyPoint> key2, map<tuple<int, int>, vector<int>>& edge_point_map) {
    vector<shared_ptr<KeyPoint>> prev_keys = get_prev_keys_of_two_key_points(key1, key2);
    vector<shared_ptr<KeyPoint>> next_keys = get_next_keys_of_two_key_points(key1, key2);

    double entropy = 0;
    int cur_num = get_key_point_merge_size(key1, key2, edge_point_map);

    for (auto& prev_key : prev_keys) {
        int edge_num = 0;
        
        auto edge_tuple1 = make_tuple(prev_key->id, key1->id);
        auto edge_tuple2 = make_tuple(prev_key->id, key2->id);

        if (edge_point_map.count(edge_tuple1)) {
            edge_num += (int)edge_point_map[edge_tuple1].size();
        }
        if (edge_point_map.count(edge_tuple2)) {
            edge_num += (int)edge_point_map[edge_tuple2].size();
        }

        double percentage = (double)edge_num / (double)cur_num;
        entropy -= percentage * log2(percentage);
    }

    for (auto& next_key : next_keys) {
        int edge_num = 0;
        
        auto edge_tuple1 = make_tuple(key1->id, next_key->id);
        auto edge_tuple2 = make_tuple(key2->id, next_key->id);

        if (edge_point_map.count(edge_tuple1)) {
            edge_num += (int)edge_point_map[edge_tuple1].size();
        }
        if (edge_point_map.count(edge_tuple2)) {
            edge_num += (int)edge_point_map[edge_tuple2].size();
        }

        double percentage = (double)edge_num / (double)cur_num;
        entropy -= percentage * log2(percentage);
    }

    return entropy;
}

void merge_key_point(shared_ptr<KeyPoint> key1, shared_ptr<KeyPoint> key2, map<tuple<int, int>, vector<int>>& edge_point_map, map<tuple<int, int>, vector2D>& chain_map, vector<shared_ptr<KeyPoint>>& key_points_group) {
    vector<shared_ptr<KeyPoint>> prev_keys = get_prev_keys_of_two_key_points(key1, key2);
    vector<shared_ptr<KeyPoint>> next_keys = get_next_keys_of_two_key_points(key1, key2);

    set<int> points;
    for (auto& point : key1->points) {
        points.insert(point->idP);
    }
    for (auto& point : key2->points) {
        points.insert(point->idP);
    }
    int cur_num = points.size();

    int sum_num = key1->points.size() + key2->points.size();
    double ratio1 = (double)key1->points.size() / (double)sum_num;
    double ratio2 = (double)key2->points.size() / (double)sum_num;

    vector1D merged_pos = ratio1 * key1->pos + ratio2 * key2->pos;

    vector1D move1 = merged_pos - key1->pos;
    vector1D move2 = merged_pos - key2->pos;

    // build merged key point
    vector<shared_ptr<ControlPoint>> new_control_points;
    for (int point : points) {
        new_control_points.push_back(shared_ptr<ControlPoint>(new ControlPoint(merged_pos, point, 0)));
    }
    auto merged_key = shared_ptr<KeyPoint>(new KeyPoint(new_control_points));

    // build edges
    for (auto& prev_key : prev_keys) {
        prev_key->DAG_edges.push_back(merged_key);
        merged_key->DAG_inv_edges.push_back(prev_key);
    }
    for (auto& next_key : next_keys) {
        merged_key->DAG_edges.push_back(next_key);
        next_key->DAG_inv_edges.push_back(merged_key);
    }

    // update chain_map
    for (auto& prev_key : prev_keys) {
        vector2D chain_prev;
        auto edge_tuple1 = make_tuple(prev_key->id, key1->id);
        auto edge_tuple2 = make_tuple(prev_key->id, key2->id);

        if (chain_map.count(edge_tuple1) && !chain_map.count(edge_tuple2)) {
            chain_prev = apply_move_on_vector2D_tail(chain_map[edge_tuple1], move1);
        } else if (!chain_map.count(edge_tuple1) && chain_map.count(edge_tuple2)) {
            chain_prev = apply_move_on_vector2D_tail(chain_map[edge_tuple2], move2);
        } else {
            vector2D chain_prev1 = apply_move_on_vector2D_tail(chain_map[edge_tuple1], move1);
            vector2D chain_prev2 = apply_move_on_vector2D_tail(chain_map[edge_tuple2], move2);
            chain_prev = merge_two_vector2D(chain_prev1, chain_prev2);
        }

        auto edge_tuple_new = make_tuple(prev_key->id, merged_key->id);
        chain_map[edge_tuple_new] = chain_prev;
    }

    for (auto& next_key : next_keys) {
        vector2D chain_next;
        auto edge_tuple1 = make_tuple(key1->id, next_key->id);
        auto edge_tuple2 = make_tuple(key2->id, next_key->id);

        if (chain_map.count(edge_tuple1) && !chain_map.count(edge_tuple2)) {
            chain_next = apply_move_on_vector2D_head(chain_map[edge_tuple1], move1);
        } else if (!chain_map.count(edge_tuple1) && chain_map.count(edge_tuple2)) {
            chain_next = apply_move_on_vector2D_head(chain_map[edge_tuple2], move2);
        } else {
            vector2D chain_next1 = apply_move_on_vector2D_head(chain_map[edge_tuple1], move1);
            vector2D chain_next2 = apply_move_on_vector2D_head(chain_map[edge_tuple2], move2);
            chain_next = merge_two_vector2D(chain_next1, chain_next2);
        }

        auto edge_tuple_new = make_tuple(merged_key->id, next_key->id);
        chain_map[edge_tuple_new] = chain_next;
    }

    // clear old key points
    for (auto& cp : key1->points) {
        cp->keypoint = nullptr;
    }
    for (auto& cp : key2->points) {
        cp->keypoint = nullptr;
    }
    key1->points.clear();
    key2->points.clear();

    // clear old edge
    for (auto& prev_key : prev_keys) {
        auto iter = find(prev_key->DAG_edges.begin(), prev_key->DAG_edges.end(), key1);
        if (iter!= prev_key->DAG_edges.end()) {
            prev_key->DAG_edges.erase(iter);
        }
        
        iter = find(prev_key->DAG_edges.begin(), prev_key->DAG_edges.end(), key2);
        if (iter!= prev_key->DAG_edges.end()) {
            prev_key->DAG_edges.erase(iter);
        }

        auto edge_tuple1 = make_tuple(prev_key->id, key1->id);
        auto edge_tuple2 = make_tuple(prev_key->id, key2->id);
        if (chain_map.count(edge_tuple1)) {
            chain_map.erase(edge_tuple1);
        }
        if (chain_map.count(edge_tuple2)) {
            chain_map.erase(edge_tuple2);
        }
    }
    key1->DAG_inv_edges.clear();
    key2->DAG_inv_edges.clear();

    auto edge_tuple12 = make_tuple(key1->id, key2->id);
    auto edge_tuple21 = make_tuple(key2->id, key1->id);
    if (chain_map.count(edge_tuple12)) {
        chain_map.erase(edge_tuple12);
    }
    if (chain_map.count(edge_tuple21)) {
        chain_map.erase(edge_tuple21);
    }

    for (auto& next_key : next_keys) {
        auto iter = find(next_key->DAG_inv_edges.begin(), next_key->DAG_inv_edges.end(), key1);
        if (iter!= next_key->DAG_inv_edges.end()) {
            next_key->DAG_inv_edges.erase(iter);
        }
        
        iter = find(next_key->DAG_inv_edges.begin(), next_key->DAG_inv_edges.end(), key2);
        if (iter!= next_key->DAG_inv_edges.end()) {
            next_key->DAG_inv_edges.erase(iter);
        }

        auto edge_tuple1 = make_tuple(key1->id, next_key->id);
        auto edge_tuple2 = make_tuple(key2->id, next_key->id);
        if (chain_map.count(edge_tuple1)) {
            chain_map.erase(edge_tuple1);
        }
        if (chain_map.count(edge_tuple2)) {
            chain_map.erase(edge_tuple2);
        }
    }
    key1->DAG_edges.clear();
    key2->DAG_edges.clear();

    // update key points group
    key_points_group.push_back(merged_key);
    key_points_group.erase(find(key_points_group.begin(), key_points_group.end(), key1));
    key_points_group.erase(find(key_points_group.begin(), key_points_group.end(), key2));
}

bool try_to_merge_key_point(const py::object& anim, shared_ptr<KeyPoint> key1, shared_ptr<KeyPoint> key2, map<tuple<int, int>, vector<int>>& edge_point_map, map<tuple<int, int>, vector2D>& chain_map, vector<shared_ptr<KeyPoint>>& key_points_group) {
    if (key1 == key2) {
        return false;
    }else if (key1->DAG_inv_edges.size() == 0 || key2->DAG_inv_edges.size() == 0) {
        return false;
    } else if (key1->DAG_edges.size() == 0 || key2->DAG_edges.size() == 0) {
        return false;
    } else {
        double entropy1 = get_single_key_point_entropy(key1, edge_point_map);
        double entropy2 = get_single_key_point_entropy(key2, edge_point_map);

        int size1 = key1->points.size();
        int size2 = key2->points.size();

        double ratio1 = (double)size1 / (double)(size1 + size2);
        double ratio2 = (double)size2 / (double)(size1 + size2);

        double before_entropy = ratio1 * entropy1 + ratio2 * entropy2;

        double after_entropy = get_key_point_merge_entropy(key1, key2, edge_point_map);

        double merge_dis = anim.attr("key_point_merge_dis").cast<double>();

        // if (after_entropy < before_entropy) {
        if (get_norm(key1->pos - key2->pos) < merge_dis) {
            // cout << "Merge key point " << key1->id << " and " << key2->id << endl;
            // cout << "    Entropy before merge: " << before_entropy << endl;
            // cout << "    Entropy after merge : " << after_entropy << endl;
            // cout << "    pos1: " << key1->pos[0] << ", " << key1->pos[1] << "\n";
            // cout << "    pos2: " << key2->pos[0] << ", " << key2->pos[1] << "\n";

            merge_key_point(key1, key2, edge_point_map, chain_map, key_points_group);
            return true;
        } else {
            return false;
        }
    }
}

bool get_merge_key_points_group(const py::object& anim, vector<shared_ptr<KeyPoint>>& key_points_group, map<tuple<int, int>, vector2D>& chain_map) {
    auto edge_point_map = get_edge_point_map(key_points_group);
    bool overall_operated = false;
    bool operated = true;
    while (operated) {
        operated = false;
        edge_point_map = get_edge_point_map(key_points_group);

        vector<tuple<shared_ptr<KeyPoint>, shared_ptr<KeyPoint>>> key_points_to_check;
        // try to merge continuous key points

        // if A->B, we try to merge A and B
        for (auto& key1 : key_points_group) {
            for (auto& key2 : key1->DAG_edges) {
                key_points_to_check.push_back(make_tuple(key1, key2));
            }
        }

        // try to merge parallel key points

        for (auto& key : key_points_group) {
            // if A->B and A->C, we try to merge B and C
            for (auto& key1 : key->DAG_edges) {
                for (auto& key2 : key->DAG_edges) {
                    key_points_to_check.push_back(make_tuple(key1, key2));
                }
            }
            
            // if B->A and C->A, we try to merge B and C
            for (auto& key1 : key->DAG_inv_edges) {
                for (auto& key2 : key->DAG_inv_edges) {
                    key_points_to_check.push_back(make_tuple(key1, key2));
                }
            }
        }

        for (auto key_point_tuple : key_points_to_check) {
            auto key1 = get<0>(key_point_tuple);
            auto key2 = get<1>(key_point_tuple);
            operated |= try_to_merge_key_point(anim, key1, key2, edge_point_map, chain_map, key_points_group);
            if (operated) {
                overall_operated = true;
                break;
            }
        }
    }

    return overall_operated;
}

struct PathPoint {
    vector1D pos;
    double timing;
    bool key;

    PathPoint(vector1D pos1,
              double timing1,
              bool key1 = false)
        : pos(pos1), timing(timing1), key(key1) {}
};

void debug_path(const vector<PathPoint>& path) {
    cout << "Debugging path: " << endl;
    for (auto& p : path) {
        cout << "    " << p.timing << " : (" << p.pos[0] << ", " << p.pos[1]
             << ") " << endl;
    }
    cout << "Over." << endl;
}

void debug_sub_cluster(const vector<vector<int>>& sub_clusters) {
    cout << "Debugging sub clusters: ";
    for (auto& sub_cluster : sub_clusters) {
        cout << "[ ";
        for (auto& id : sub_cluster) {
            cout << id << " ";
        }
        cout << "]  ";
    }
    cout << endl;
}

double get_stretch_ratio(const char* ease_function) {
    if (strcmp(ease_function, "Linear") == 0) {
        return 1;
    } else {
        auto pytweening = py::module::import("pytweening");
        string ease_out_func_str = string("easeOut").append(ease_function);
        const char* ease_out_func = ease_out_func_str.c_str();

        double step = 1e-5;
        double value = pytweening.attr(ease_out_func)(step).cast<double>();
        double stretch_ratio = value / step;

        return stretch_ratio;
    }
}

double apply_ease(double cur_time,
                  double start_time,
                  double end_time,
                  double ease_ratio,
                  const char* ease_function) {
    double sum_time = end_time - start_time;
    double t = (cur_time - start_time) / sum_time;

    // tranform t according to ease function
    if (strcmp(ease_function, "Linear") == 0) {
        t = t;
    } else if (ease_ratio < t && t < 1 - ease_ratio) {
        t = t;
    } else {
        // we only ease the start and end part of the curve

        auto pytweening = py::module::import("pytweening");
        string ease_out_func_str = string("easeOut").append(ease_function);
        string ease_in_func_str = string("easeIn").append(ease_function);
        const char* ease_out_func = ease_out_func_str.c_str();
        const char* ease_in_func = ease_in_func_str.c_str();

        double stretch_ratio = get_stretch_ratio(ease_function);
        double eased_start = ease_ratio * (1 - stretch_ratio);
        // double eased_end = 1 - ease_ratio * (1 - stretch_ratio);

        if (t < 0.5) {
            t = (t - eased_start) / (ease_ratio * stretch_ratio);
            t = max(0.0, t);
            t = pytweening.attr(ease_in_func)(t).cast<double>();
            t = t * ease_ratio;
        } else {
            t = (t - (1 - ease_ratio)) / (ease_ratio * stretch_ratio);
            t = min(1.0, t);
            t = pytweening.attr(ease_out_func)(t).cast<double>();
            t = (1 - ease_ratio) + t * ease_ratio;
        }
        // string ease_in_out_func_str =
        // string("easeInOut").append(ease_function); const char*
        // ease_in_out_func = ease_in_out_func_str.c_str(); t =
        // pytweening.attr(ease_in_out_func)(t).cast<double>();
    }

    double eased_time = start_time + sum_time * t;
    return eased_time;
}

tuple<double, double> get_eased_start_end(double start_time,
                                          double end_time,
                                          double ease_ratio,
                                          const char* ease_function) {
    double stretch_ratio = get_stretch_ratio(ease_function);
    double sum_time = end_time - start_time;

    double eased_start_t = ease_ratio * (1 - stretch_ratio);
    double eased_end_t = 1 - ease_ratio * (1 - stretch_ratio);

    double eased_start = start_time + eased_start_t * sum_time;
    double eased_end = start_time + eased_end_t * sum_time;
    return {eased_start, eased_end};
}

// return the position at the given timing
vector1D interpolate_path_by_time(const vector<PathPoint>& path,
                                  double timing) {
    // apply tweening
    // auto pytweening = py::module::import("pytweening");
    // timing = pytweening.attr(ease_function)(timing).cast<double>();

    int s = (int)path.size();

    if (s == 0) {
        return {0, 0};
    } else if (timing < path[0].timing) {
        return path[0].pos;
    } else if (timing >= path[s - 1].timing) {
        return path[s - 1].pos;
    }

    for (int i = 0; i < s - 1; i++) {
        if (path[i].timing <= timing && timing < path[i + 1].timing) {
            double t = (timing - path[i].timing) /
                       (path[i + 1].timing - path[i].timing);
            return path[i].pos + t * (path[i + 1].pos - path[i].pos);
        }
    }

    return path[s - 1].pos;
}



void get_topo_sort_key_points(vector<shared_ptr<KeyPoint>>& key_points) {
    // get in deg
    map<int, int> deg;
    for (auto cur : key_points) {
        deg[cur->id] = (int)cur->DAG_inv_edges.size();
    }

    // init queue
    queue<shared_ptr<KeyPoint>> q;
    for (auto cur : key_points) {
        if (deg[cur->id] == 0) {
            q.push(cur);
        }
    }

    // begin topo sort
    vector<int> id_list;
    while (!q.empty()) {
        auto cur = q.front();
        q.pop();
        id_list.push_back(cur->id);

        for (auto next : cur->DAG_edges) {
            deg[next->id]--;
            if (deg[next->id] == 0) {
                q.push(next);
            }
        }
    }

    map<int, int> rank;
    for (int i = 0; i < (int)id_list.size(); i++) {
        rank[id_list[i]] = i;
    }

    // check whether all key points ranked
    for (auto cur : key_points) {
        if (rank.count(cur->id) == 0) {
            cout << "Warning! KeyPoint " << cur->id << " not ranked" << endl;
            rank[cur->id] = (int)id_list.size();
        }
    }

    // sort the origin key points by rank
    sort(key_points.begin(), key_points.end(),
         [&](shared_ptr<KeyPoint>& a, shared_ptr<KeyPoint>& b) {
             return rank[a->id] < rank[b->id];
         });
}

tuple<vector<vector<shared_ptr<ControlPoint>>>, vector<shared_ptr<KeyPoint>>>
get_key_points(const py::object& anim, const vector<int>& group, const vector<vector<int>>& sub_groups) {
    vector1D radii = anim.attr("radii").cast<vector1D>();
    double radius = anim.attr("r").cast<double>();
    radii *= radius;

    vector3D control_points = anim.attr("control_points").cast<vector3D>();
    
    // compute main_dir
    vector1D avg_start(2, 0);
    vector1D avg_end(2, 0);
    for (int i : group) {
        int N_control_points = (int)control_points[i].size();
        avg_start += control_points[i][0];
        avg_end += control_points[i][N_control_points - 1];
    }
    avg_start /= (double)group.size();
    avg_end /= (double)group.size();
    vector1D main_dir = get_unit_vector(avg_end - avg_start);

    // remove reverse
    for (auto& sub_group : sub_groups) {
        auto sub_group_CPs = control_points[sub_group[0]];

        // vector1D avg_start_sub_group = sub_group_CPs[0];
        // vector1D avg_end_sub_group = sub_group_CPs[sub_group_CPs.size() - 1];
        vector1D avg_start_sub_group(2, 0);
        vector1D avg_end_sub_group(2, 0);
        for (int i : sub_group) {
            avg_start_sub_group += control_points[i][0];
            avg_end_sub_group += control_points[i][control_points[i].size() - 1];
        }
        avg_start_sub_group /= (double)sub_group.size();
        avg_end_sub_group /= (double)sub_group.size();

        sub_group_CPs.erase(sub_group_CPs.begin());
        sub_group_CPs.erase(sub_group_CPs.end() - 1);

        int N_control_points = (int)sub_group_CPs.size();

        auto check_reverse = [&](const vector1D& pos1, const vector1D& pos2, const vector1D& dir, double dot_value) {
            return vector_dot(get_unit_vector(pos2 - pos1), dir) < dot_value;
        };

        auto check_start_reverse = [&](const vector1D& pos) {
            vector1D dir = get_unit_vector(pos - avg_start_sub_group);
            for (int i : sub_group) {
                vector1D start_pos = control_points[i][0];
                if (check_reverse(start_pos, pos, dir, 0.5)) {
                    return true;
                }
            }
            return false;
        };

        auto check_end_reverse = [&](const vector1D& pos) {
            vector1D dir = get_unit_vector(avg_end_sub_group - pos);
            for (int i : sub_group) {
                vector1D end_pos = control_points[i][control_points[i].size() - 1];
                if (check_reverse(pos, end_pos, dir, 0.5)) {
                    return true;
                }
            }
            return false;
        };

        // remove start reverse
        for (int j = 0; j < N_control_points; j++) {
            if (check_start_reverse(sub_group_CPs[j])) {
                sub_group_CPs.erase(sub_group_CPs.begin() + j);
                N_control_points--;
                j--;
            } else {
                double left = 0;
                double right = 1;

                auto get_pos = [&](double t) {
                    return (1 - t) * avg_start_sub_group + t * sub_group_CPs[j];
                };

                while (right - left > 1e-3) {
                    double t = (left + right) / 2;
                    vector1D cur_pos_t = get_pos(t);
                    if (check_start_reverse(cur_pos_t)) {
                        left = t;
                    } else {
                        right = t;
                    }
                }

                vector1D final_pos = get_pos(right);
                sub_group_CPs.insert(sub_group_CPs.begin(), final_pos);
                N_control_points++;
                break;
            }
        }

        // // remove mid reverse
        for (int j = 1; j < N_control_points; j++) {
            if (check_reverse(sub_group_CPs[j - 1], sub_group_CPs[j], main_dir, 0.5)) {
                sub_group_CPs.erase(sub_group_CPs.begin() + j);
                N_control_points--;
                j--;
            }
        }

        // remove end reverse
        for (int j = N_control_points - 1; j >= 0; j--) {
            if (check_end_reverse(sub_group_CPs[j])) {
                sub_group_CPs.erase(sub_group_CPs.begin() + j);
                N_control_points--;
            } else {
                double left = 0;
                double right = 1;

                auto get_pos = [&](double t) {
                    return (1 - t) * avg_end_sub_group + t * sub_group_CPs[j];
                };

                while (right - left > 1e-3) {
                    double t = (left + right) / 2;
                    vector1D cur_pos_t = get_pos(t);
                    if (check_end_reverse(cur_pos_t)) {
                        left = t;
                    } else {
                        right = t;
                    }
                }

                vector1D final_pos = get_pos(right);
                sub_group_CPs.push_back(final_pos);
                N_control_points++;
                break;
            }
        }

        // rebuild all control points
        for (int i : sub_group) {
            vector1D start_pos = control_points[i][0];
            vector1D end_pos = control_points[i][control_points[i].size() - 1];
            
            control_points[i] = {start_pos};
            control_points[i].insert(control_points[i].end(), sub_group_CPs.begin(), sub_group_CPs.end());
            control_points[i].push_back(end_pos);
        }
    }
        
    // double merge_dis = anim.attr("key_point_merge_dis").cast<double>();
    int N = (int)control_points.size();

    vector<vector<shared_ptr<ControlPoint>>> control_points_group;
    vector<shared_ptr<KeyPoint>> key_points_group;

    for (int i = 0; i < N; i++) {
        control_points_group.push_back(vector<shared_ptr<ControlPoint>>());
    }

    vector<shared_ptr<ControlPoint>> all_control_points;

    for (int i : group) {
        int N_control_points = (int)control_points[i].size();
        for (int u = 0; u < N_control_points; u++) {
            vector1D pos = control_points[i][u];
            auto new_cp = shared_ptr<ControlPoint>(
                new ControlPoint(pos, i, u));
            control_points_group[i].push_back(new_cp);
            all_control_points.push_back(new_cp);
        }
    }

    UFS ufs(global_control_point_num);
    vector<bool> unioned(global_control_point_num, false);

    // find key points as a bundle of control points
    for (int i : group) {
        int N_control_points_i = (int)control_points[i].size();
        // auto start_pos_i = control_points_group[i][0]->pos;
        // auto end_pos_i =
        //             control_points_group[i][N_control_points_i - 1]->pos;
        // auto dir_i = get_unit_vector(end_pos_i - start_pos_i);

        for (int u = 1; u < N_control_points_i - 1; u++) {
            auto cpiu = control_points_group[i][u];
            if (unioned[cpiu->idU]) {
                continue;
            }
            unioned[cpiu->idU] = true;

            for (int j : group) {      
                if (i == j) {
                    continue;
                }
                
                int N_control_points_j = (int)control_points[j].size();
                // auto start_pos_j = control_points_group[j][0]->pos;
                // auto end_pos_j =
                //     control_points_group[j][N_control_points_j - 1]->pos;
                // auto dir_j = get_unit_vector(end_pos_j - start_pos_j);

                // if (vector_dot(dir_i, dir_j) < 0) {
                //     continue;
                // }

                for (int v = 1; v < N_control_points_j - 1; v++) {
                    auto cpjv = control_points_group[j][v];
                    if (unioned[cpjv->idU]) {
                        continue;
                    }

                    double dist = get_norm(cpiu->pos - cpjv->pos);
                
                    if (dist < 1e-5) {
                        ufs.union_set(cpiu->idU, cpjv->idU);
                        unioned[cpjv->idU] = true;
                    }
                }
            }
        }
    }

    auto bundled_control_points = ufs.get_group();

    int N_all_key_points = (int)bundled_control_points.size();
    for (int i = 0; i < N_all_key_points; i++) {
        vector<shared_ptr<ControlPoint>> cur_bundled_control_points;
        for (int idU : bundled_control_points[i]) {
            cur_bundled_control_points.push_back(all_control_points[idU]);
            auto cp = all_control_points[idU];
        }
        auto new_key = shared_ptr<KeyPoint>(new KeyPoint(
            cur_bundled_control_points));
        key_points_group.push_back(new_key);
        for (auto& p : cur_bundled_control_points) {
            p->keypoint = new_key;
        }
    }

    return {control_points_group, key_points_group};
}

void check_valid_control_paths(const vector<int>& group, const vector<vector<shared_ptr<ControlPoint>>>& control_points_group){
    for (int i : group) {
        int N_control_points = (int)control_points_group[i].size();
        for (int u = 0; u < N_control_points - 1; u++) {
            auto cur = control_points_group[i][u];
            auto next = control_points_group[i][u + 1];
            auto cur_key = cur->keypoint;
            auto next_key = next->keypoint;
            // check whether the control points are in the same key point or there is an edge from cur to next
            if (cur_key->id != next_key->id && find(cur_key->DAG_edges.begin(), cur_key->DAG_edges.end(), next_key) == cur_key->DAG_edges.end()) {
                cout<<"warning: no edges from "<<cur_key->id<<" to "<<next_key->id<<endl;
            }
        }
    }
}

void get_DAG_of_key_points(
    const py::object& anim, 
    const vector<int>& group,
    vector<vector<shared_ptr<ControlPoint>>>& control_points_group,
    vector<shared_ptr<KeyPoint>>& key_points_group) {
    // we need to make sure there is no loop in the DAG
    int N_KP = (int)key_points_group.size();
    double start_padding_dis = anim.attr("start_padding_dis").cast<double>();

    vector<vector<bool>> access(N_KP, vector<bool>(N_KP, false));
    for (int key_id = 0; key_id < N_KP; key_id++) {
        access[key_id][key_id] = true;
    }

    // compute main_dir
    // vector1D avg_start(2, 0);
    // vector1D avg_end(2, 0);
    // for (int i : group) {
    //     int N_control_points = (int)control_points_group[i].size();
    //     avg_start += control_points_group[i][0]->pos;
    //     avg_end += control_points_group[i][N_control_points - 1]->pos;
    // }
    // avg_start /= (double)group.size();
    // avg_end /= (double)group.size();
    // vector1D main_dir = get_unit_vector(avg_end - avg_start);

    for (int i : group) {
        int N_control_points = (int)control_points_group[i].size();
        auto prev_key = control_points_group[i][0]->keypoint;
        bool isStart = true;

        for (int u = 1; u < N_control_points; u++) {
            shared_ptr<KeyPoint> cur_key = control_points_group[i][u]->keypoint;
            if (cur_key == nullptr) {
                cout<<"warning: keyIU is nullptr. "<<"point: "<<i<<" "<<u<<endl;
                continue;
            }

            if (cur_key == prev_key) {
                continue;
            }

            bool need_to_jump = false;

            if (access[cur_key->id][prev_key->id]) {
                // Detect loop
                // cout << "Remove loop from path " << i << endl;
                need_to_jump = true;
            } else if (isStart && u < N_control_points - 1 && get_norm(prev_key->pos - cur_key->pos) < start_padding_dis) {
                // cout << "Jump Start " << i << endl;
                need_to_jump = true;
            } 
            // else if (vector_dot(get_unit_vector(cur_key->pos - prev_key->pos), main_dir) < 0.5 && u < N_control_points - 1) {
            //     cout << "Jump " << i << " to avoid reverse edge" << endl;
            //     need_to_jump = true;
            // } 

            if (need_to_jump) {
                // we need to drop the control point in the cur keypoint
                auto iter = find(cur_key->points.begin(), cur_key->points.end(), control_points_group[i][u]);
                if (iter == cur_key->points.end()) {
                    cout << "error: cannot find control point in keypoint " << cur_key->id << endl;
                }
                cur_key->points.erase(iter);
                control_points_group[i][u]->keypoint = nullptr;
            } else {
                isStart = false;

                vector<int> prev_access;
                for (int other_key_id = 0; other_key_id < N_KP; other_key_id++) {
                    if (access[other_key_id][prev_key->id]) {
                        prev_access.push_back(other_key_id);
                    }
                }
                vector<int> next_access;
                for (int other_key_id = 0; other_key_id < N_KP; other_key_id++) {
                    if (access[cur_key->id][other_key_id]) {
                        next_access.push_back(other_key_id);
                    }
                }
                for (int prev_other_key_id : prev_access) {
                    for (int next_other_key_id : next_access) {
                        access[prev_other_key_id][next_other_key_id] = true;
                    }
                }
                prev_key->DAG_edges.push_back(cur_key);
                cur_key->DAG_inv_edges.push_back(prev_key);
                prev_key = cur_key;
            }
        }
    }

    // remove the repeated edges
    for (auto key : key_points_group) {
        set<shared_ptr<KeyPoint>> set_edges(key->DAG_edges.begin(),
                                            key->DAG_edges.end());
        key->DAG_edges.assign(set_edges.begin(), set_edges.end());

        set<shared_ptr<KeyPoint>> set_inv_edges(key->DAG_inv_edges.begin(),
                                                key->DAG_inv_edges.end());
        key->DAG_inv_edges.assign(set_inv_edges.begin(), set_inv_edges.end());
    }

    // remove empty control points
    for (int i : group) {
        vector<shared_ptr<ControlPoint>> new_control_points;
        for (auto cp : control_points_group[i]) {
            if (cp->keypoint!= nullptr) {
                new_control_points.push_back(cp);
            }
        }
        control_points_group[i] = new_control_points;
    }

    // remove empty key points
    vector<shared_ptr<KeyPoint>> key_points_to_remove;
    for (auto key : key_points_group) {
        if (key->points.empty()) {
            key_points_to_remove.push_back(key);
        }
    }

    for (auto key : key_points_to_remove) {
        auto iter = find(key_points_group.begin(), key_points_group.end(), key);
        if (iter!= key_points_group.end()) {
            key_points_group.erase(iter);
            // cout << "Remove empty key point " << key->id << endl;
        } else {
            cout << "error: cannot find keypoint " << key->id << endl;
        }
    }
}

bool get_simplified_DAG(vector<shared_ptr<KeyPoint>>& key_points_group, map<tuple<int, int>, vector2D>& chain_map) {
    bool operated = false;
    while (true) {
        vector<int> id_to_remove;

        for (auto& key : key_points_group) {
            int deg_in = (int) key->DAG_inv_edges.size();
            int deg_out = (int) key->DAG_edges.size();
            if (deg_in == 1 && deg_out == 1) {
                // this key point can be removed
                auto& front_key = key->DAG_inv_edges[0];
                auto& back_key = key->DAG_edges[0];

                // update the chain map

                auto edge_tuple_front = make_tuple(front_key->id, key->id);
                auto edge_tuple_back = make_tuple(key->id, back_key->id);

                auto chain_front = chain_map[edge_tuple_front];
                auto chain_back = chain_map[edge_tuple_back];

                chain_map.erase(edge_tuple_front);
                chain_map.erase(edge_tuple_back);

                auto chain_combined = chain_front;
                chain_combined.insert(chain_combined.end(), chain_back.begin() + 1, chain_back.end()); // +1 for ignore start

                auto edge_tuple_combined = make_tuple(front_key->id, back_key->id);
                if (chain_map.count(edge_tuple_combined)) {
                    chain_combined = merge_two_vector2D(chain_map[edge_tuple_combined], chain_combined);
                }
                chain_map[edge_tuple_combined] = chain_combined;

                // update the DAG

                auto iter_front = find(front_key->DAG_edges.begin(),
                                       front_key->DAG_edges.end(), key);
                auto iter_back = find(back_key->DAG_inv_edges.begin(),
                                      back_key->DAG_inv_edges.end(), key);

                auto iter_front_back =
                    find(front_key->DAG_edges.begin(),
                         front_key->DAG_edges.end(), back_key);

                auto iter_back_front =
                    find(back_key->DAG_inv_edges.begin(),
                         back_key->DAG_inv_edges.end(), front_key);

                if (iter_front_back == front_key->DAG_edges.end()) {
                    *iter_front = back_key;
                } else {
                    front_key->DAG_edges.erase(iter_front);
                }

                if (iter_back_front == back_key->DAG_inv_edges.end()) {
                    *iter_back = front_key;
                } else {
                    back_key->DAG_inv_edges.erase(iter_back);
                }

                id_to_remove.push_back(key->id);
                for (auto cp : key->points) {
                    cp->keypoint = nullptr;
                }
            }
        }

        for (int id : id_to_remove) {
            int loc = -1;
            int len = (int)key_points_group.size();
            for (int i = 0; i < len; i++) {
                if (key_points_group[i]->id == id) {
                    loc = i;
                    break;
                }
            }
            key_points_group.erase(key_points_group.begin() + loc);
        }

        if (id_to_remove.size() == 0) {
            break;
        } else {
            operated = true;
        }
    }

    return operated;
}

map<int, double> get_key_point_slow_rate(const vector<shared_ptr<KeyPoint>>& key_points_group) {
    map<int, double> key_point_slow_rate;

    auto edge_point_map = get_edge_point_map(key_points_group);

    for (auto& key : key_points_group) {
        int cur_num = (int)key->points.size();
        double slow_rate = 1;
        double slow_num = get_single_key_point_entropy(key, edge_point_map);

        slow_rate = 1 + slow_num * log(cur_num);
        key_point_slow_rate[key->id] = slow_rate;
    }
    // scale the slow rate into [1, 2]
    double min_slow_rate = 1;
    double max_slow_rate = 1;
    for (auto& key : key_points_group) {
        min_slow_rate = min(min_slow_rate, key_point_slow_rate[key->id]);
        max_slow_rate = max(max_slow_rate, key_point_slow_rate[key->id]);
    }
    for (auto& key : key_points_group) {
        key_point_slow_rate[key->id] = 1 + (key_point_slow_rate[key->id] - min_slow_rate) / (max_slow_rate - min_slow_rate);
    }

    return key_point_slow_rate;
}

bool remove_DAG_triangle_edge(vector<shared_ptr<KeyPoint>>& key_points_group, map<tuple<int, int>, vector2D>& chain_map) {
    // triangle : A->B, B->C, A->C
    // we remove A->C
    // first, we must know which points are on the edge
    auto edge_point_map = get_edge_point_map(key_points_group);

    // cout << "compute edge points\n";

    vector<tuple<shared_ptr<KeyPoint>, shared_ptr<KeyPoint>>> edges_to_remove;

    for (auto& keyA : key_points_group) {
        for (auto& keyB : keyA->DAG_edges) {
            for (auto& keyC : keyB->DAG_edges) {
                if (keyA->id == keyC->id || keyB->id == keyC->id ||
                    keyA->id == keyB->id) {
                    continue;
                }

                auto iter =
                    find(keyA->DAG_edges.begin(), keyA->DAG_edges.end(), keyC);
                if (iter == keyA->DAG_edges.end()) {
                    // there is no edge A->C
                    continue;
                }

                auto edge_tuple_AC = make_tuple(keyA->id, keyC->id);
                auto edge_tuple_BC = make_tuple(keyB->id, keyC->id);
                auto edge_tuple_AB = make_tuple(keyA->id, keyB->id);

                if (edge_point_map.count(edge_tuple_AC) == 0 ||
                    edge_point_map.count(edge_tuple_AB) == 0 ||
                    edge_point_map.count(edge_tuple_BC) == 0) {
                    // A->C or A->B or B->C already removed
                    continue;
                }
                // A->B, B->C, A->C

                // cout << "remove triangle : " << keyA->id << "-" << keyB->id
                //      << "-" << keyC->id << endl;

                // first, we delete A->C and its inverse edge
                edges_to_remove.push_back(make_tuple(keyA, keyC));

                // then, we put the point on A->C to A->B, B->C
                vector<int> edge_point_AC = edge_point_map[edge_tuple_AC];
                vector<int> edge_point_AB = edge_point_map[edge_tuple_AB];
                vector<int> edge_point_BC = edge_point_map[edge_tuple_BC];

                vector<int> unioned_edge_point_AB;
                vector<int> unioned_edge_point_BC;

                set_union(edge_point_AB.begin(), edge_point_AB.end(),
                          edge_point_AC.begin(), edge_point_AC.end(),
                          inserter(unioned_edge_point_AB,
                                   unioned_edge_point_AB.begin()));
                set_union(edge_point_BC.begin(), edge_point_BC.end(),
                          edge_point_AC.begin(), edge_point_AC.end(),
                          inserter(unioned_edge_point_BC,
                                   unioned_edge_point_BC.begin()));

                edge_point_map.erase(edge_tuple_AC);
                edge_point_map[edge_tuple_AB] = unioned_edge_point_AB;
                edge_point_map[edge_tuple_BC] = unioned_edge_point_BC;

                // cout << "put the point on A->C to A->B, B->C" << endl;

                // finally, we create new control points in B
                for (int point : edge_point_AC) {
                    // ignore idC and idU
                    auto cp = shared_ptr<ControlPoint>(new ControlPoint(
                        keyB->pos, point, 0));
                    cp->keypoint = keyB;
                    keyB->points.push_back(cp);
                }

                // cout << "create new control points in B" << endl;

                // move B to the mass center
                vector1D mass_center = (keyA->pos + keyB->pos + keyC->pos) / 3;

                vector1D move = mass_center - keyB->pos;
                keyB->pos = mass_center;
                
                // update the chain
                chain_map.erase(edge_tuple_AC);
            
                for (auto prev_key : keyB->DAG_inv_edges) {
                    auto edge_tuple = make_tuple(prev_key->id, keyB->id);
                    if (chain_map.count(edge_tuple) > 0) {
                        chain_map[edge_tuple] = apply_move_on_vector2D_tail(chain_map[edge_tuple], move);
                    }
                }
                for (auto next_key : keyB->DAG_edges) {
                    auto edge_tuple = make_tuple(keyB->id, next_key->id);
                    if (chain_map.count(edge_tuple) > 0) {
                        chain_map[edge_tuple] = apply_move_on_vector2D_head(chain_map[edge_tuple], move);
                    }
                }
            }
        }
    }

    // cout << "found triangle done\n";

    for (auto& edge : edges_to_remove) {
        auto& keyA = get<0>(edge);
        auto& keyC = get<1>(edge);
        // cout<<"remove edge "<<keyA->id<<"->"<<keyC->id<<endl;
        auto iterAC =
            find(keyA->DAG_edges.begin(), keyA->DAG_edges.end(), keyC);
        auto iterCA =
            find(keyC->DAG_inv_edges.begin(), keyC->DAG_inv_edges.end(), keyA);
        if (iterAC == keyA->DAG_edges.end() || iterCA == keyC->DAG_inv_edges.end()) {
            cout<<"error: edge not found"<<endl;
        }
        else{
            keyA->DAG_edges.erase(iterAC);
            keyC->DAG_inv_edges.erase(iterCA);        
        }
    }

    if (edges_to_remove.size()!= 0) {
        return true;
    } else {
        return false;
    }
}

vector2D get_clamped_cubic_spline(const vector2D& origin_path, int inter_num, const vector1D& start_dir={0, 0}, const vector1D& end_dir={0, 0}) {
    py::object interface_module = py::module::import("utils.interpolate.interpolate");
    py::object interface_func = interface_module.attr("get_clamped_cubic_spline");
    vector2D smoothed_path = interface_func(origin_path, inter_num, start_dir, end_dir).cast<vector2D>();
    return smoothed_path;
}

map<tuple<int, int>, vector2D> get_smooth_chain_map(const py::object& anim, const vector<int>& group, vector<shared_ptr<KeyPoint>>& key_points_group) {
    // first, we need to extract each point's path
    int N = anim.attr("N").cast<int>();
    int inter_num = anim.attr("inter_num").cast<int>();

    vector3D origin_paths(N);
    vector<int> N_paths(N, 0);
    map<tuple<int, int>, int> key_points_index_for_point;

    for (auto key : key_points_group) {
        auto points = key->getClusterVector();
        for (int p : points) {
            origin_paths[p].push_back(key->pos);
            key_points_index_for_point[make_tuple(key->id, p)] = N_paths[p]++;
        }
    }

    // then, we interpolate the path for each point
    vector3D interpolated_paths(N);
    for (int p : group) {
        interpolated_paths[p] = get_clamped_cubic_spline(origin_paths[p], inter_num);
    }

    // we need to know which points are on the edge
    auto edge_point_map = get_edge_point_map(key_points_group);
    map<tuple<int, int>, vector2D> smooth_chain_map;

    // finnaly, we interpolate the whole DAG
    for (auto key : key_points_group) {
        for (auto next_key : key->DAG_edges) {

            // then we get the points on the edge
            vector<int> points = edge_point_map[make_tuple(key->id, next_key->id)];
            int N_points = (int)points.size();

            vector2D smooth_chain;
        
            for (int i = 0; i <= inter_num; i++) {
                vector1D average_pos({0, 0});

                for (int p : points) {
                    // get current pos on interpolated path
                    int cur_index = key_points_index_for_point[make_tuple(key->id, p)];
                    int cur_inter_index = cur_index * inter_num + i;
                    vector1D pos = interpolated_paths[p][cur_inter_index];
                    average_pos += pos;
                }

                average_pos = average_pos / (double)N_points;

                smooth_chain.push_back(average_pos);
            }

            smooth_chain_map[make_tuple(key->id, next_key->id)] = smooth_chain;
        }
    }

    return smooth_chain_map;
}

void convert_deg2_to_chain(vector<shared_ptr<KeyPoint>>& key_points_group, map<tuple<int, int>, vector2D>& smooth_chain_map) {
    auto chain_map_origin = smooth_chain_map;
    smooth_chain_map.clear();
    vector<shared_ptr<KeyPoint>> key_points_remove;

    for (auto key : key_points_group) {
        if (key->DAG_edges.size() == 1 && key->DAG_inv_edges.size() == 1) {
            continue;
        }

        auto origin_edges = key->DAG_edges;
        for (auto next_key : origin_edges) {
            vector2D chain = chain_map_origin[make_tuple(key->id, next_key->id)];
            while (next_key->DAG_edges.size() == 1 && next_key->DAG_inv_edges.size() == 1) {
                key_points_remove.push_back(next_key);
                auto next_next_key = next_key->DAG_edges[0];
                vector2D next_chain = chain_map_origin[make_tuple(next_key->id, next_next_key->id)];

                // ignore the start
                chain.insert(chain.end(), next_chain.begin() + 1, next_chain.end());
                next_key = next_next_key;
            }

            if (find(key->DAG_edges.begin(), key->DAG_edges.end(), next_key) == key->DAG_edges.end()) {
                key->DAG_edges.push_back(next_key);
                next_key->DAG_inv_edges.push_back(key);
            }
            smooth_chain_map[make_tuple(key->id, next_key->id)] = chain;
        }
    }

    // clear
    for (auto key_to_remove : key_points_remove) {
        auto iter = find(key_points_group.begin(), key_points_group.end(), key_to_remove);
        if (iter == key_points_group.end()) {
            cout << "Warning: KeyPoint " << key_to_remove->id << " is not in the key_points_group \n";
            continue;
        }
        key_points_group.erase(iter);
    }

    // clear edges
    for (auto key_to_remove : key_points_remove) {
        for (auto key : key_points_group) {
            auto iter = find(key->DAG_edges.begin(), key->DAG_edges.end(), key_to_remove);
            if (iter!= key->DAG_edges.end()) {
                key->DAG_edges.erase(iter);
            }
            iter = find(key->DAG_inv_edges.begin(), key->DAG_inv_edges.end(), key_to_remove);
            if (iter!= key->DAG_inv_edges.end()) {
                key->DAG_inv_edges.erase(iter);
            }
        }
    }
}

void check_DAG_legal(vector<shared_ptr<KeyPoint>>& key_points_group) {
    for (auto& key : key_points_group) {
        auto cur_cluster = key->getClusterVector();
        for (auto& next_key : key->DAG_edges) {
            auto next_cluster = next_key->getClusterVector();
            for (auto point : next_cluster) {
                if (find(cur_cluster.begin(), cur_cluster.end(), point) !=
                    cur_cluster.end()) {
                    auto iter = find(cur_cluster.begin(), cur_cluster.end(), point);
                    cur_cluster.erase(iter);
                }
            }
        }
        if (cur_cluster.size() > 0 && key->DAG_edges.size() > 0) {
            cout << "Error! The key point " << key->id
                 << " has points not in the next key point " << endl;
        }
        cur_cluster = key->getClusterVector();
        for (auto& prev_key : key->DAG_inv_edges) {
            auto prev_cluster = prev_key->getClusterVector();
            for (auto point : prev_cluster) {
                if (find(cur_cluster.begin(), cur_cluster.end(), point) !=
                    cur_cluster.end()) {
                    // remove the point from cur_cluster
                    auto iter = find(cur_cluster.begin(), cur_cluster.end(), point);
                    cur_cluster.erase(iter);
                }
            }
        }
        if (cur_cluster.size() > 0 && key->DAG_inv_edges.size() > 0) {
            cout << "Error! The key point " << key->id
                 << " has points not in the prev key point " << endl;
        }
    }
}

void check_edge_dir(vector<shared_ptr<KeyPoint>> & key_points_group){
    map<int,int> id_to_order = {};
    for (int i = 0; i < key_points_group.size(); i++) {
        id_to_order[key_points_group[i]->id] = i;
    }
    for (auto & key : key_points_group){
        for (auto & next_key : key->DAG_edges){
            if (id_to_order[next_key->id] < id_to_order[key->id]){
                cout<<"Error! The edge "<<key->id<<"->"<<next_key->id<<" is in wrong direction"<<endl;
            }
        }
        for (auto & prev_key : key->DAG_inv_edges){
            if (id_to_order[prev_key->id] > id_to_order[key->id]){
                cout<<"Error! The edge "<<prev_key->id<<"->"<<key->id<<" is in wrong direction"<<endl;
            }
        }
    }
}

tuple<double, double> get_min_max_timing(
    const vector<vector<PathPoint>>& paths) {
    double min_timing = numeric_limits<double>::max();
    double max_timing = numeric_limits<double>::min();
    for (auto& path : paths) {
        for (auto& point : path) {
            if (point.timing < min_timing) {
                min_timing = point.timing;
            }
            if (point.timing > max_timing) {
                max_timing = point.timing;
            }
        }
    }
    return {min_timing, max_timing};
}

struct PackedKeyPoint {
    vector1D pos;
    double timing;
    double timing_pre;
    double timing_fin;
    double merge_time;
    double split_time;

    vector2D end_merge_delays;
    vector2D start_merge_delays;
    vector2D start_split_delays;
    vector2D end_split_delays;

    vector2D end_merge_pos;
    vector2D start_merge_pos;
    vector2D start_split_pos;
    vector2D end_split_pos;

    map<int, int> point_froms_cc_order;
    map<int, int> point_nexts_cc_order;

    int stay = 0;

    vector1D avg_end_pos;
    vector1D avg_start_pos;

    vector<vector<int>> sub_clusters;
    vector<int> full_sub_cluster;
    vector1D packing_radii;
    vector2D packing_pos;
    vector2D next_sub_cluster_centers;
    vector2D prev_sub_cluster_centers;
    vector<int> froms;
    vector<int> nexts;
    vector1D froms_packing_radius;
    vector1D nexts_packing_radius;
    int id;
};

vector2D get_incremental_packing_python(
    const int n_pre,
    const vector<vector2D>& pre_packing_pos,
    const vector<vector1D>& pre_avg_end_pos,
    const vector<vector1D>& pre_avg_start_pos,
    const vector<vector1D>& pre_radius,
    const vector<vector<vector<int>>>& pre_sub_clusters,
    const vector<vector<int>>& cur_sub_clusters,
    bool global_pack,
    vector<bool> local_pack,
    double contour_width) {
    // auto start_time = std::chrono::high_resolution_clock::now();
    py::object interface_module =
        py::module::import("utils.powerdiagramPacking.interface");
    py::object interface = interface_module.attr("new_pd_packing_interface");
    vector2D result_pos = pre_packing_pos[0];
    map<int,int> id_to_pre;
    for (int i = 0; i < pre_sub_clusters.size(); i++) {
        for (int j = 0; j < pre_sub_clusters[i].size(); j++) {
            for (int k = 0; k < pre_sub_clusters[i][j].size(); k++) {
                id_to_pre[pre_sub_clusters[i][j][k]] = i;
            }
        }
    }
    for (int i=0;i<result_pos.size();i++){
        result_pos[i][0] = pre_packing_pos[id_to_pre[i]][i][0];
        result_pos[i][1] = pre_packing_pos[id_to_pre[i]][i][1];
    }


    double time_pd = 0;
    double time_fd = 0;
    double time_gb = 0;

    // local pack stage
    for (int i=0;i<cur_sub_clusters.size();i++){
        // cout<<"i = "<<i<<" local_pack = "<<local_pack[i]<<endl;
        if (local_pack[i]){
            vector<int> filtered_cur_sub_cluster = cur_sub_clusters[i];
            vector<vector<vector<int>>> filtered_pre_sub_clusters = {};
            vector<vector2D> filtered_pre_packing_pos = {};
            vector<vector1D> filtered_pre_avg_end_pos = {};
            vector<vector1D> filtered_pre_avg_start_pos = {};
            vector<vector1D> filtered_pre_radius = {};
            
            for (int j=0;j<pre_sub_clusters.size();j++){
                vector<vector<int>> filtered_pre_sub_cluster = {};
                for (int k=0;k<pre_sub_clusters[j].size();k++){
                    vector<int> filtered_pre_sub_cluster_k = {};
                    for (int l=0;l<pre_sub_clusters[j][k].size();l++){
                        if (find(cur_sub_clusters[i].begin(), cur_sub_clusters[i].end(), pre_sub_clusters[j][k][l]) != cur_sub_clusters[i].end()){
                            filtered_pre_sub_cluster_k.push_back(pre_sub_clusters[j][k][l]);
                        }
                    }
                    if (filtered_pre_sub_cluster_k.size() > 0){
                        filtered_pre_sub_cluster.push_back(filtered_pre_sub_cluster_k);
                    }
                }
                if (filtered_pre_sub_cluster.size() > 0){
                    filtered_pre_sub_clusters.push_back(filtered_pre_sub_cluster);
                    filtered_pre_packing_pos.push_back(pre_packing_pos[j]);
                    filtered_pre_avg_end_pos.push_back(pre_avg_end_pos[j]);
                    filtered_pre_avg_start_pos.push_back(pre_avg_start_pos[j]);
                    filtered_pre_radius.push_back(pre_radius[j]);
                }
            }
            auto start_time = chrono::high_resolution_clock::now();

            tuple<vector2D,vector<int>, vector<vector<int>>> results = 
            interface(filtered_pre_sub_clusters.size(), filtered_pre_packing_pos, filtered_pre_avg_end_pos, filtered_pre_avg_start_pos,
                    filtered_pre_radius, filtered_pre_sub_clusters, filtered_cur_sub_cluster)
                .cast<tuple<vector2D,vector<int>,vector<vector<int>>>>();
            auto end_time = chrono::high_resolution_clock::now();
            time_pd += chrono::duration_cast<chrono::nanoseconds>(end_time-start_time).count() / 1000000000.0;

            vector2D local_result_pos = get<0>(results);
            // cout<<"result_pos.size() = "<<local_result_pos.size()<<endl;
            vector<int> result_sub_cluster = get<1>(results);
            if (result_sub_cluster.size() == 0) {
                return local_result_pos;
            }
            vector<int> reverse_result_sub_cluster = vector<int>(pre_packing_pos[0].size(), -1);
            for (int j = 0; j < result_sub_cluster.size(); j++) {
                reverse_result_sub_cluster[result_sub_cluster[j]] = j;
            }
            vector<vector<int>> result_attraction_pairs = get<2>(results);
            int n = (int) result_sub_cluster.size();
            double* positions = new double[n * 2];
            double* radii = new double[n];
            int n_attraction_pairs = (int) result_attraction_pairs.size();
            int* attraction_pairs = new int[n_attraction_pairs * 2];
            for (int j = 0; j < n; j++) {
                positions[j * 2] = local_result_pos[result_sub_cluster[j]][0];
                positions[j * 2 + 1] = local_result_pos[result_sub_cluster[j]][1];
                radii[j] = pre_radius[0][result_sub_cluster[j]];
            }
            for (int j = 0; j < n_attraction_pairs; j++) {
                attraction_pairs[j * 2] = result_attraction_pairs[j][0];
                attraction_pairs[j * 2 + 1] = result_attraction_pairs[j][1];
            }
            double size_mag = 200;
            double gravity_mag = 1e3;
            double attraction_mag = 70.0;
            int n_iters = 200;
            double alpha_min = 0.1;

            start_time = chrono::high_resolution_clock::now();
            vector2D res_positions = Simulate(n, positions, radii, n_attraction_pairs,
                                            attraction_pairs, size_mag, gravity_mag,
                                            attraction_mag, n_iters, alpha_min);
            end_time = chrono::high_resolution_clock::now();
            time_fd += chrono::duration_cast<chrono::nanoseconds>(end_time-start_time).count() / 1000000000.0;
            // make center to 0, 0
            vector1D center(2, 0);
            for (int j = 0; j < n; j++) {
                center += res_positions[j];
            }
            center /= (double)n;
            for (int j = 0; j < n; j++) {
                res_positions[j] -= center;
            }

            // make center to avg end
            center = vector1D(2, 0);
            for (int j = 0; j < (int)filtered_pre_avg_end_pos.size(); j++) {
                center += filtered_pre_avg_end_pos[j];
            }
            center /= (double)filtered_pre_avg_end_pos.size();
            for (int j = 0; j < n; j++) {
                res_positions[j] += center;
            }

            // std::cout<<"res_positions.size() = "<<res_positions.size()<<std::endl;
            for (int j = 0; j < n; j++) {
                // cout<<"circle "<<result_sub_cluster[j]<<" "<<res_positions[j][0]<<" "<<res_positions[j][1]<<endl;
                result_pos[result_sub_cluster[j]][0] = res_positions[j][0];
                result_pos[result_sub_cluster[j]][1] = res_positions[j][1];
            }
            
        }
        else
        {
            for (int j = 0; j < cur_sub_clusters[i].size() ; j++) {
                int point = cur_sub_clusters[i][j];
                int cur_id_to_pre = id_to_pre[point];

                // cout<<"circle "<<point<<" "<<pre_packing_pos[cur_id_to_pre][point][0]<<" "<<pre_packing_pos[cur_id_to_pre][point][1]<<endl;
                result_pos[point] = pre_packing_pos[cur_id_to_pre][point] + pre_avg_end_pos[cur_id_to_pre];
                // result_pos[point] = pre_packing_pos[cur_id_to_pre][point];
            }
        }
    }

    if (global_pack) {
        // cout << "GlobalPacking" << endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        result_pos = GlobalPackingBubbleTree(result_pos, pre_radius[0],
                                             cur_sub_clusters, contour_width);
        auto end_time = std::chrono::high_resolution_clock::now();
        time_gb += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / 1000000000.0;
        // cout << "GlobalPacking Done" << endl;
    }

    // cout << "        time_pd: " << time_pd << endl;
    // cout << "        time_fd: " << time_fd << endl;
    // cout << "        time_gb: " << time_gb << endl;
    return result_pos;
}

vector<PackedKeyPoint> get_packing_of_key_points(
    const py::object& anim,
    const vector<shared_ptr<KeyPoint>>& key_points_group,
    const vector<vector<shared_ptr<ControlPoint>>>& control_points_group,
    int group_id) {
    vector<PackedKeyPoint> packed_key_points;
    vector<vector<vector<int>>> sub_groups_per_group = anim.attr("sub_groups_per_group").cast<vector<vector<vector<int>>>>();
    vector<vector<int>> sub_groups = sub_groups_per_group[group_id];
    map<int, int> id_map;
    int N_key_points = (int)key_points_group.size();
    // cout << "N_key_points: " << N_key_points << endl;
    int N = (int)control_points_group.size();

    double radius_factor = anim.attr("r").cast<double>();
    radius_factor *= 1.1;
    vector1D radii = anim.attr("radii").cast<vector1D>();

    auto start0 = std::chrono::high_resolution_clock::now();
    // construct id_map
    for (int i = 0; i < N_key_points; i++) {
        id_map[key_points_group[i]->id] = i;
    }

    for (int i = 0; i < N_key_points; i++) {
        PackedKeyPoint packed_key_point;
        packed_key_point.pos = key_points_group[i]->pos;
        packed_key_point.id = key_points_group[i]->id;
        // packed_key_point.sub_cluster = key_points_group[i]->getClusterVector();

        // cout<<"key_points_group[i]->id = "<<key_points_group[i]->id<<endl;
        // cout<<"key_points_group[i]->pos = "<<key_points_group[i]->pos[0]<<", "<<key_points_group[i]->pos[1]<<endl;
        // TODO
        packed_key_point.sub_clusters = key_points_group[i]->getSubClusterVector(sub_groups);
        packed_key_point.full_sub_cluster = key_points_group[i]->getClusterVector();
        
        
        // get from
        for (auto kp : key_points_group[i]->DAG_inv_edges) {
            packed_key_point.froms.push_back(id_map[kp->id]);
        }

        // get next
        for (auto kp : key_points_group[i]->DAG_edges) {
            packed_key_point.nexts.push_back(id_map[kp->id]);
        }

        // init packing pos
        for (int j = 0; j < N; j++) {
            packed_key_point.packing_pos.push_back({0, 0});
        }
        // init packing radii
        for (int j = 0; j < N; j++) {
            packed_key_point.packing_radii.push_back(radii[j]*radius_factor);
        }

        packed_key_points.push_back(packed_key_point);
    }

    auto edge_point_map = get_edge_sub_cluster_map(key_points_group, sub_groups);


    int reverse_packing = 1;
    vector<int> order;
    for (int i = 0; i < N_key_points; i++) {
        order.push_back(i);
    }
    vector<int> reverse_order = order;
    reverse(reverse_order.begin(), reverse_order.end());
    if (reverse_packing) {
        order = reverse_order;
    }

    for (int i = 0; i < N_key_points; i++) {
        auto cur = packed_key_points[i];
        if (cur.nexts.size()==0 && cur.froms.size()>=1){
            order.push_back(i);
        }
    }
    vector<int> packed_flags(N_key_points, 0);
    auto end0 = std::chrono::high_resolution_clock::now();
    double data_prepare_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0)
            .count()/1000.0;
    // compute all the packing layout
    double packing_time = 0;
    double packing_algorithm_time = 0;
    for (int i : order) {
        auto start = std::chrono::high_resolution_clock::now();
        if (packed_flags[i] == 1) {
            continue;
        }
        auto& cur = packed_key_points[i];
        
        bool global_pack = true;
        vector<bool> local_pack(cur.sub_clusters.size(), false);
        if (cur.full_sub_cluster.size() <= 1) {
            cur.packing_pos[cur.full_sub_cluster[0]] = vector1D({0, 0});
            packed_flags[i] = 1;
            continue;
        } else if (cur.froms.size() == 0 || cur.nexts.size() == 0) {
            packed_flags[i] = 1;
            for (auto CP : key_points_group[i]->points) {
                cur.packing_pos[CP->idP] = CP->pos - key_points_group[i]->pos;
                // cout<<"cur.packing_pos[CP->idP] = "<<cur.packing_pos[CP->idP][0]<<", "<<cur.packing_pos[CP->idP][1]<<endl;
            }
            continue;
        }
        // merge or split
        vector<PackedKeyPoint> pres;
        // vector<int> raw_index;
        if (reverse_packing) {
            for (int j : cur.nexts) {
                pres.push_back(packed_key_points[j]);
                // raw_index.push_back(j);
            }
        } else {
            for (int j : cur.froms) {
                pres.push_back(packed_key_points[j]);
                // raw_index.push_back(j);
            }
        }

        vector<vector2D> pre_packing_pos;
        vector<vector1D> pre_avg_end_pos;
        vector<vector1D> pre_avg_start_pos;
        vector<vector1D> pre_radius;
        vector<vector<vector<int>>> pre_sub_clusters;

        // for further support non-uniform circles
        int n_circles = (int)pres[0].packing_pos.size();
        vector1D all_radii = get_zero1D(n_circles);
        for (int j = 0; j < n_circles; j++) {
            all_radii[j] += pres[0].packing_radii[j];
        }

        if (reverse_packing) {
            if (cur.nexts.size() == 1) {
                auto& pre = pres[0];
                if (pre.froms.size() > 1) {
                    pre_packing_pos.push_back(pre.packing_pos);
                    pre_avg_end_pos.push_back(pre.pos);
                    pre_avg_start_pos.push_back(pre.avg_start_pos);
                    pre_radius.push_back(all_radii);
                    pre_sub_clusters.push_back(cur.sub_clusters);
                    local_pack[0] = true;
                    cur.stay = 1;
                    global_pack = true;
                } else {
                    cur.packing_pos = pre.packing_pos;
                    packed_flags[i] = 1;
                    continue;
                }
            } else {
                for (auto& pre : pres) {
                    // because reverse, cur -> pre
                    auto sub_clusters = edge_point_map[make_tuple(cur.id, pre.id)];
                    if (sub_clusters.size() == 0) {
                        // cout << "sub_cluster is empty" << endl;
                        // cout << "  cur.id = " << cur.id << endl;
                        // cout << "  pre.id = " << pre.id << endl;
                        sub_clusters = edge_point_map[make_tuple(pre.id, cur.id)];
                    }
                    pre_packing_pos.push_back(pre.packing_pos);
                    pre_avg_end_pos.push_back(pre.pos);
                    pre_avg_start_pos.push_back(pre.avg_start_pos);
                    pre_radius.push_back(all_radii);
                    pre_sub_clusters.push_back(sub_clusters);
                    global_pack = true;
                }
            }
        } else {
            if (cur.froms.size() == 1) {
                auto& pre = pres[0];
                if (pre.nexts.size() > 1) {
                    pre_packing_pos.push_back(pre.packing_pos);
                    pre_avg_end_pos.push_back(pre.avg_end_pos);
                    pre_avg_start_pos.push_back(pre.avg_start_pos);
                    pre_radius.push_back(all_radii);
                    pre_sub_clusters.push_back(cur.sub_clusters);
                } else {
                    cur.packing_pos = pre.packing_pos;
                }
            } else {
                for (auto& pre : pres) {
                    while ( pre.froms.size() == 1 &&
                            pre.nexts.size() == 1) {
                        pre = packed_key_points[pre.froms[0]];
                    }
                    pre_packing_pos.push_back(pre.packing_pos);
                    pre_avg_end_pos.push_back(pre.avg_end_pos);
                    pre_avg_start_pos.push_back(pre.avg_start_pos);
                    pre_radius.push_back(all_radii);
                    pre_sub_clusters.push_back(pre.sub_clusters);
                }
            }
        }

        if (pre_packing_pos.size() >= 1) {
            auto start_packing = std::chrono::high_resolution_clock::now();
            
            int n_pres_key_points = (int)pre_packing_pos.size();
            vector<set<int>> cur_sub_cluster_set;
            vector<set<int>> pre_sub_cluster_set;
            for (int j = 0; j < cur.sub_clusters.size(); j++) {
                cur_sub_cluster_set.push_back(set<int>(cur.sub_clusters[j].begin(), cur.sub_clusters[j].end()));
            }
            for (int j = 0; j < pre_sub_clusters.size(); j++) {
                for (int k = 0; k < pre_sub_clusters[j].size(); k++) {
                    pre_sub_cluster_set.push_back(set<int>(pre_sub_clusters[j][k].begin(), pre_sub_clusters[j][k].end()));
                }
            }
            bool all_in = true;
            for (int j = 0; j < cur.sub_clusters.size(); j++) {
                bool in = false;
                for (int k = 0; k < pre_sub_cluster_set.size(); k++) {
                    if (cur_sub_cluster_set[j] == pre_sub_cluster_set[k]) {
                        in = true;
                        break;
                    }
                }
                if (!in) {
                    all_in = false;
                    local_pack[j] = true;
                    cur.stay = 1;
                    // break;
                }
            }

            vector<vector<int>> cur_sub_clusters = cur.sub_clusters;

            cur.packing_pos = get_incremental_packing_python(
                (int)pre_packing_pos.size(), pre_packing_pos,
                pre_avg_end_pos, pre_avg_start_pos, pre_radius, 
                pre_sub_clusters,cur_sub_clusters, global_pack, local_pack, radius_factor / 4);
            // cout<<"Packing Over!"<<endl;
            auto end_packing = std::chrono::high_resolution_clock::now();
            packing_algorithm_time +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_packing - start_packing)
                    .count() /
                1000.0;
            ////

            // make center to (0, 0)
            vector1D center({0, 0});
            for (int point : cur.full_sub_cluster) {
                center = center + cur.packing_pos[point];
            }
            int N_sub_cluster = (int)cur.full_sub_cluster.size();
            center = center / N_sub_cluster;
            // cout << "Packing pos " << i << " : \n";
            for (int point : cur.full_sub_cluster) {
                cur.packing_pos[point] =
                    cur.packing_pos[point] - center;
                
                // cout << "  " << point << ": " << cur.packing_pos[point][0] << ", " << cur.packing_pos[point][1] << "\n";
            }
        }
        packed_flags[i] = 1;

        if (cur.nexts.size() == 0) {
            int from = cur.froms[0];
            while(packed_key_points[from].froms.size() == 1 && packed_key_points[from].nexts.size() == 1){
                packed_key_points[from].packing_pos = cur.packing_pos;
                from = packed_key_points[from].froms[0];
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        // double cur_packing_time =
        //     std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        //         .count();
        // cout<<"cur_packing_time = "<<cur_packing_time<<" ms"<<endl;
        packing_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1000.0;
    }

    // cout<<"    data_prepare_time = "<<data_prepare_time<<endl;
    // cout<<"    inner_packing_time = "<<packing_time<<endl;
    // cout<<"    inner_packing_algorithm_time = "<<packing_algorithm_time<<endl;
    return packed_key_points;
}

vector<vector<vector<int>>> sub_cluster_division(
    const vector<vector<int>>& sub_clusters,
    const vector3D& sub_cluster_packing_pos,
    const vector<int>& cur_sub_cluster,
    const vector2D& cur_packing_pos,
    const vector1D& radii,
    const vector2D& directions) {
    // use python interface to calculate sub_cluster division
    py::object interface_module =
        py::module::import("utils.powerdiagramPacking.interface");
    py::object interface = interface_module.attr("sub_cluster_division");
    vector<vector<vector<int>>> result =
        interface(sub_clusters, sub_cluster_packing_pos, cur_sub_cluster,
                  cur_packing_pos, radii, directions)
            .cast<vector<vector<vector<int>>>>();
    return result;
}

vector<PathPoint> get_smooth_velocity_path(const vector2D& trajectory, double head_timing, double tail_timing, double head_velocity, double tail_velocity) {
    double total_length = get_vector2D_length(trajectory);
    double total_t = tail_timing - head_timing;

    // cout << "l : " << total_length << endl;
    // cout << "t : " << tail_timing - head_timing << endl;
    // cout << "k1: " << head_velocity << endl;
    // cout << "k2: " << tail_velocity << endl;

    vector1D BS_start({head_timing, 0});
    vector1D BS_end({tail_timing, total_length});

    double time_step = total_t / 2;

    vector1D BS_control_start({head_timing + time_step, head_velocity * time_step});
    vector1D BS_control_end({tail_timing - time_step, total_length - tail_velocity * time_step});

    if (head_velocity * time_step > total_length) {
        double cur_time_step = total_length / head_velocity;
        BS_control_start = {head_timing + cur_time_step, head_velocity * cur_time_step};
    }

    if (tail_velocity * time_step > total_length) {
        double cur_time_step = total_length / tail_velocity;
        BS_control_end = {tail_timing - cur_time_step, total_length - tail_velocity * cur_time_step};
    }

    
    vector2D BS_controls({BS_control_start, BS_control_end});

    py::object interface = py::module::import("utils.interpolate.interpolate");
    py::object inter_func = interface.attr("get_bspline_curve");

    vector2D BS_curve = inter_func(BS_start, BS_controls, BS_end).cast<vector2D>();


    auto get_inter_pathpoint = [&](double x, double t) {
        x = min(x, total_length);
        x = max(x, 0.0);

        int N_path = (int)trajectory.size();
        for (int i = 0; i < N_path - 1; i++) {
            double cur_length = get_norm(trajectory[i] - trajectory[i + 1]);
            if (x < cur_length) {
                double rate = x / cur_length;
                vector1D pos = (1 - rate) * trajectory[i] + rate * trajectory[i + 1];
                return PathPoint(pos, t);
            } else {
                x -= cur_length;
            }
        }

        return PathPoint(trajectory[N_path - 1], t);
    };

    vector<PathPoint> result;

    int N_sample = (int) BS_curve.size();
    for (int i = 0; i < N_sample; i++) {
        double t = BS_curve[i][0];
        double x = BS_curve[i][1];

        result.push_back(get_inter_pathpoint(x, t));
    }

    return result;
}

// generate packing path
tuple<vector<vector<PathPoint>>, vector<vector<PathPoint>>> get_packing_path(
    const py::object& anim,
    const vector<int>& group,
    // const vector<vector<PathPoint>>& paths,
    const vector<vector<shared_ptr<ControlPoint>>>& control_points_group,
    const vector<shared_ptr<KeyPoint>>& key_points_group,
    vector<PackedKeyPoint>& packed_key_points,
    map<tuple<int, int>, vector2D>& smooth_chain_map, int group_id) {

    int N_key_points = (int)key_points_group.size();
    // cout<<"N_key_points: "<<N_key_points<<endl;
    int N = (int)control_points_group.size();

    vector<vector<vector<int>>> sub_groups_per_group = anim.attr("sub_groups_per_group").cast<vector<vector<vector<int>>>>();
    vector<vector<int>> sub_groups = sub_groups_per_group[group_id];

    double radius = anim.attr("r").cast<double>();
    // radius *= 1.1;

    // slow down top k entropy
    int slowdown_topk = (int)(anim.attr("slowdown_topk").cast<double>() * N_key_points);
    slowdown_topk = max(slowdown_topk, 1);
    
    auto key_point_slow_rate = get_key_point_slow_rate(key_points_group);
    map<int, double> real_slow_rate;
    vector<int> key_points_ids;
    for (int i = 0; i < N_key_points; i++) {
        key_points_ids.push_back(i);
        real_slow_rate[key_points_group[i]->id] = 1;
    }
    sort(key_points_ids.begin(), key_points_ids.end(), [&](int a, int b) {
        int id_a = key_points_group[a]->id;
        int id_b = key_points_group[b]->id;
        return key_point_slow_rate[id_a] > key_point_slow_rate[id_b];
    });
    slowdown_topk = min(slowdown_topk, (int)key_points_ids.size());
    

    for (int i = 0; i < slowdown_topk; i++) {
        int id = key_points_ids[i];
        int real_id = key_points_group[id]->id;
        real_slow_rate[real_id] = key_point_slow_rate[real_id];

        // cout << "SlowRate" << i << ": " << real_slow_rate[real_id] << endl;
    }

    // convert packing result to path
    // vector<vector<PathPoint>> packing_paths;
    vector<vector<PathPoint>> packing_pos_paths;
    vector<vector<PathPoint>> center_paths;


    // init the path and push back the start point
    for (int i = 0; i < N; i++) {
        // packing_paths.push_back(vector<PathPoint>());
        packing_pos_paths.push_back(vector<PathPoint>());
        center_paths.push_back(vector<PathPoint>());
    }


    double base_velocity = anim.attr("base_velocity").cast<double>();
    double trans_speed_rate = anim.attr("trans_speed").cast<double>();
    double sedimentation_speed_rate = anim.attr("sedimentation_speed").cast<double>();

    double trans_velocity = base_velocity * trans_speed_rate;
    double sedimentation_velocity = base_velocity * sedimentation_speed_rate;

    auto edge_point_map = get_edge_point_map(key_points_group);
    auto edge_sub_cluster_map = get_edge_sub_cluster_map(key_points_group, sub_groups);

    map<tuple<int, int>, double> edge_head_cut_require_map;
    map<tuple<int, int>, double> edge_tail_cut_require_map;
    map<tuple<int, int>, vector2D> smooth_chain_with_center_map;
    map<tuple<int, int>, vector2D> smooth_chain_cut_map;
    map<tuple<int,int>, double> edge_costs;
    map<tuple<int, int>, int> N_components_map;
    map<tuple<int, int>, int> split_N_components_map;


    for (int i = 0; i < N_key_points; i++) {
        auto& cur = packed_key_points[i];
        cur.next_sub_cluster_centers =
            vector2D(cur.packing_pos.size(), vector1D({0, 0}));
        cur.prev_sub_cluster_centers =
            vector2D(cur.packing_pos.size(), vector1D({0, 0}));
    }

    // cout << "rua0\n";

    for (int i = 0; i < N_key_points; i++) {
        auto& cur = packed_key_points[i];

        int N_froms = (int)cur.froms.size();
        int N_nexts = (int)cur.nexts.size();

        double cur_slow_rate = real_slow_rate[cur.id];

        if (N_froms > 1) {
            map<int, int> point_froms_order;

            for (int j = 0; j < N_froms; j++) {
                int from = cur.froms[j];
                auto& pre = packed_key_points[from];

                auto edge_tuple = make_tuple(pre.id, cur.id);
                auto sub_clusters = edge_sub_cluster_map[edge_tuple];

                for (auto sub_cluster : sub_clusters) {
                    for (int point : sub_cluster) {
                        point_froms_order[point] = j;
                    }
                }
            }
            vector1D main_dir({0, 0});
            for (int j = 0; j < N_froms; j++) {
                auto pre = packed_key_points[cur.froms[j]];
                main_dir = main_dir + cur.pos + cur.packing_pos[j] - pre.pos - pre.packing_pos[j];
            }
            main_dir = get_unit_vector(main_dir);

            vector3D froms_sub_cluster_packing_pos(N_froms);
            vector<vector<vector<int>>> froms_sub_clusters(N_froms);

            for (int j = 0; j < N_froms; j++) {
                auto& pre = packed_key_points[cur.froms[j]];
                auto edge_tuple = make_tuple(pre.id, cur.id);

                froms_sub_cluster_packing_pos[j] = pre.packing_pos;
                froms_sub_clusters[j] = edge_sub_cluster_map[edge_tuple];
            }

            vector1D radii = cur.packing_radii;
            vector2D directions = vector2D(radii.size(), vector1D({0, 0}));
            for (int j = 0; j < radii.size(); j++) {
                auto pre = packed_key_points[cur.froms[point_froms_order[j]]];
                directions[j] = cur.pos + cur.packing_pos[j] - pre.pos - pre.packing_pos[j];
                directions[j] = get_unit_vector(directions[j]);
            }
            vector<vector<vector<int>>> sub_cluster_connected_components = vector<vector<vector<int>>>(0);
            for (int j = 0; j < N_froms; j++) {
                vector<vector<int>> sub_cluster_connected_component;
                for (int k = 0; k < froms_sub_clusters[j].size(); k++) {
                    sub_cluster_connected_component.push_back(froms_sub_clusters[j][k]);
                }
                sub_cluster_connected_components.push_back(sub_cluster_connected_component);
            }

            map<int, int> point_froms_cc_order;
            for (int j = 0; j < N_froms; j++) {
                for (int k = 0; k < sub_cluster_connected_components[j].size();
                     k++) {
                    for (int point : sub_cluster_connected_components[j][k]) {
                        point_froms_cc_order[point] = k;
                    }
                }
            }
            int Ncc = 0;  // N connected components
            for (int j = 0; j < N_froms; j++) {
                Ncc += (int)sub_cluster_connected_components[j].size();
            }
            

            vector2D sub_cluster_centers = vector2D(N_froms, vector1D({0, 0}));
            vector<int> sub_cluster_sizes = vector<int>(N_froms, 0);
            for (int point : cur.full_sub_cluster) {
                int from_order = point_froms_order[point];
                sub_cluster_sizes[from_order] += 1;
                sub_cluster_centers[from_order] =
                    sub_cluster_centers[from_order] + cur.packing_pos[point];
            }

            for (int j = 0; j < N_froms; j++) {
                sub_cluster_centers[j] =
                    sub_cluster_centers[j] / (double)sub_cluster_sizes[j];
            }

            vector3D sub_cluster_cc_centers = vector3D(N_froms, vector2D());
            for (int j = 0; j < N_froms; j++) {
                sub_cluster_cc_centers[j] =
                    vector2D(sub_cluster_connected_components[j].size(),
                             vector1D({0, 0}));
            }
            vector<vector<int>> sub_cluster_cc_sizes =
                vector<vector<int>>(N_froms, vector<int>());
            for (int j = 0; j < N_froms; j++) {
                sub_cluster_cc_sizes[j] =
                    vector<int>(sub_cluster_connected_components[j].size(), 0);
            }
            for (int point : cur.full_sub_cluster) {
                int from_order = point_froms_order[point];
                vector<vector<int>> ccs =
                    sub_cluster_connected_components[from_order];
                for (int k = 0; k < ccs.size(); k++) {
                    if (find(ccs[k].begin(), ccs[k].end(), point) !=
                        ccs[k].end()) {
                        sub_cluster_cc_sizes[from_order][k] += 1;
                        sub_cluster_cc_centers[from_order][k] =
                            sub_cluster_cc_centers[from_order][k] +
                            cur.packing_pos[point];
                        break;
                    }
                }
            }

            for (int j = 0; j < N_froms; j++) {
                for (int k = 0; k < sub_cluster_connected_components[j].size();
                     k++) {
                    sub_cluster_cc_centers[j][k] =
                        sub_cluster_cc_centers[j][k] /
                        (double)sub_cluster_cc_sizes[j][k];
                }
            }

            for (int point : cur.full_sub_cluster) {
                int from_order = point_froms_order[point];
                cur.prev_sub_cluster_centers[point] =
                    sub_cluster_centers[from_order];
            }

            vector2D sub_cluster_cc_center_projections =
                vector2D(N_froms, vector1D());
            for (int j = 0; j < N_froms; j++) {
                sub_cluster_cc_center_projections[j] =
                    vector1D(sub_cluster_connected_components[j].size(), 0);
            }
            double min_projection = numeric_limits<double>::max();

            for (int j = 0; j < N_froms; j++) {
                for (int k = 0; k < sub_cluster_connected_components[j].size();
                     k++) {
                    sub_cluster_cc_center_projections[j][k] =
                        vector_dot(sub_cluster_cc_centers[j][k], main_dir);
                    min_projection =
                        min(min_projection,
                            sub_cluster_cc_center_projections[j][k]);
                }
            }

            vector2D end_delays(N_froms, vector1D());

            for (int j = 0; j < N_froms; j++) {
                int from = cur.froms[j];
                auto& pre = packed_key_points[from];

                for (int k = 0; k < sub_cluster_connected_components[j].size();
                     k++) {
                    end_delays[j].push_back(
                        (sub_cluster_cc_center_projections[j][k] - min_projection) / (sedimentation_velocity / cur_slow_rate));
                }

                N_components_map[make_tuple(pre.id, cur.id)] = (int)sub_cluster_connected_components[j].size();
            }

            cur.end_merge_delays = end_delays;
            cur.point_froms_cc_order = point_froms_cc_order;

            // vector<double> instability_factors = vector<double>(N_froms, 0);
            vector3D cur_sub_cluster_packing_pos =
                vector3D(N_froms, vector2D());
            vector3D pre_sub_cluster_packing_pos =
                vector3D(N_froms, vector2D());

            for (int point : cur.full_sub_cluster) {
                int from_order = point_froms_order[point];
                int from = cur.froms[from_order];
                cur_sub_cluster_packing_pos[from_order].push_back(
                    cur.packing_pos[point]);
                pre_sub_cluster_packing_pos[from_order].push_back(
                    packed_key_points[from].packing_pos[point]);
            }

            vector1D packing_radius = vector1D(N_froms, 0);

            for (int j = 0; j < N_froms; j++) {

                double max_dist = 0;
                for (int k=0; k<pre_sub_cluster_packing_pos[j].size();k++){
                    max_dist = max(max_dist, get_norm(pre_sub_cluster_packing_pos[j][k])+radius);
                }

                packing_radius[j] = max_dist;
            }

            cur.froms_packing_radius = packing_radius;
        }

        if (N_nexts > 1) {
            vector<double> next_delays = vector<double>(N_nexts, 0);
            vector<double> proj_main_dirs = vector<double>(N_nexts, 0);
            map<int, int> point_nexts_order;

            for (int j = 0; j < N_nexts; j++) {
                int next = cur.nexts[j];
                auto& nxt = packed_key_points[next];

                auto edge_tuple = make_tuple(cur.id, nxt.id);
                // auto points = edge_point_map[edge_tuple];
                auto sub_clusters = edge_sub_cluster_map[edge_tuple];
                for (auto sub_cluster : sub_clusters) {
                    for (int point : sub_cluster) {
                        point_nexts_order[point] = j;
                    }
                }
            }
            vector1D main_dir({0, 0});
            for (int j = 0; j < N_nexts; j++) {
                auto nxt = packed_key_points[cur.nexts[j]];
                main_dir = main_dir + nxt.pos + nxt.packing_pos[j] - cur.pos - cur.packing_pos[j];
            }
            main_dir = get_unit_vector(main_dir);

            // cout<<"main_dir: "<<main_dir[0]<<" "<<main_dir[1]<<endl;
            // vectro3D nexts_sub_cluster_packing_pos(N_nexts);
            vector<vector<vector<int>>> nexts_sub_clusters(N_nexts);
            for (int j = 0; j < N_nexts; j++) {
                auto& nxt = packed_key_points[cur.nexts[j]];
                auto edge_tuple = make_tuple(cur.id, nxt.id);
                // nexts_sub_cluster_packing_pos[j] = nxt.packing_pos;
                nexts_sub_clusters[j] = edge_sub_cluster_map[edge_tuple];
            }
            // cout<<"nexts_sub_clusters size: "<<nexts_sub_clusters.size()<<endl;
            
            vector<vector<vector<int>>> sub_cluster_connected_components = vector<vector<vector<int>>>(0);
            for (int j = 0; j < N_nexts; j++) {
                vector<vector<int>> sub_cluster_connected_component;
                for (int k = 0; k < nexts_sub_clusters[j].size(); k++) {
                    sub_cluster_connected_component.push_back(nexts_sub_clusters[j][k]);
                }
                sub_cluster_connected_components.push_back(sub_cluster_connected_component);
            }
            // cout<<"sub_cluster_connected_components size: "<<sub_cluster_connected_components.size()<<endl;
            map<int, int> point_nexts_cc_order;
            for (int j = 0; j < N_nexts; j++) {
                for (int k = 0; k < sub_cluster_connected_components[j].size();
                     k++) {
                    for (int point : sub_cluster_connected_components[j][k]) {
                        point_nexts_cc_order[point] = k;
                    }
                }
            }

            int Ncc = 0;  // N connected components
            for (int j = 0; j < N_nexts; j++) {
                Ncc += (int)sub_cluster_connected_components[j].size();
            }

            // cout<<"Ncc: "<<Ncc<<endl;

            vector2D sub_cluster_centers = vector2D(N_nexts, vector1D({0, 0}));
            vector<int> sub_cluster_sizes = vector<int>(N_nexts, 0);

            for (int point : cur.full_sub_cluster) {
                int next_order = point_nexts_order[point];
                sub_cluster_sizes[next_order] += 1;
                sub_cluster_centers[next_order] =
                    sub_cluster_centers[next_order] + cur.packing_pos[point];

            }
            for (int j = 0; j < N_nexts; j++) {
                sub_cluster_centers[j] =
                    sub_cluster_centers[j] / (double)sub_cluster_sizes[j];
            }

            vector3D sub_cluster_cc_centers = vector3D(N_nexts, vector2D());
            for (int j = 0; j < N_nexts; j++) {
                sub_cluster_cc_centers[j] =
                    vector2D(sub_cluster_connected_components[j].size(),
                             vector1D({0, 0}));
            }
            vector<vector<int>> sub_cluster_cc_sizes =
                vector<vector<int>>(N_nexts, vector<int>());
            for (int j = 0; j < N_nexts; j++) {
                sub_cluster_cc_sizes[j] =
                    vector<int>(sub_cluster_connected_components[j].size(), 0);
            }
            for (int point : cur.full_sub_cluster) {
                int next_order = point_nexts_order[point];
                vector<vector<int>> ccs =
                    sub_cluster_connected_components[next_order];
                for (int k = 0; k < ccs.size(); k++) {
                    if (find(ccs[k].begin(), ccs[k].end(), point) !=
                        ccs[k].end()) {
                        sub_cluster_cc_sizes[next_order][k] += 1;
                        sub_cluster_cc_centers[next_order][k] =
                            sub_cluster_cc_centers[next_order][k] +
                            cur.packing_pos[point];
                        break;
                    }
                }
            }
            for (int j = 0; j < N_nexts; j++) {
                for (int k = 0; k < sub_cluster_connected_components[j].size();
                     k++) {
                    sub_cluster_cc_centers[j][k] =
                        sub_cluster_cc_centers[j][k] /
                        (double)sub_cluster_cc_sizes[j][k];
                }
            }

            for (int point : cur.full_sub_cluster) {
                int next_order = point_nexts_order[point];
                cur.next_sub_cluster_centers[point] =
                    sub_cluster_centers[next_order];
            }

            // vector<double> sub_cluster_center_projections =
            //     vector<double>(N_nexts, 0);
            // double max_projection = numeric_limits<double>::min();
            vector2D sub_cluster_cc_center_projections =
                vector2D(N_nexts, vector1D());
            for (int j = 0; j < N_nexts; j++) {
                sub_cluster_cc_center_projections[j] =
                    vector1D(sub_cluster_connected_components[j].size(), 0);
            }
            double max_projection = numeric_limits<double>::min();

            // for (int j = 0; j < N_nexts; j++) {
            //     sub_cluster_center_projections[j] =
            //         vector_dot(sub_cluster_centers[j], main_dir);
            //     max_projection =
            //         max(max_projection, sub_cluster_center_projections[j]);
            // }
            for (int j = 0; j < N_nexts; j++) {
                for (int k = 0; k < sub_cluster_connected_components[j].size();
                     k++) {
                    sub_cluster_cc_center_projections[j][k] =
                        vector_dot(sub_cluster_cc_centers[j][k], main_dir);
                    max_projection = max(max_projection,
                                        sub_cluster_cc_center_projections[j][k]);
                }
            }

            // cout<<"max_projection: "<<max_projection<<endl;
            // vector<double> start_delays = vector<double>(N_nexts, 0);
            vector2D start_delays = vector2D(N_nexts, vector1D());

            // for (int j = 0; j < N_nexts; j++) {
            //     start_delays[j] =
            //         (max_projection - sub_cluster_center_projections[j]) / (sedimentation_velocity / cur_slow_rate);
            // }
            for (int j = 0; j < N_nexts; j++) {
                int next = cur.nexts[j];
                auto& nxt = packed_key_points[next];
                for (int k = 0; k < sub_cluster_connected_components[j].size();
                     k++) {
                    start_delays[j].push_back(
                        (max_projection - sub_cluster_cc_center_projections[j][k]) / (sedimentation_velocity / cur_slow_rate));
                }
                split_N_components_map[make_tuple(cur.id, nxt.id)] = (int)sub_cluster_connected_components[j].size();
            }

            cur.start_split_delays = start_delays;
            cur.point_nexts_cc_order = point_nexts_cc_order;

            // vector<double> instability_factors = vector<double>(N_nexts, 0);
            vector3D cur_sub_cluster_packing_pos =
                vector3D(N_nexts, vector2D());
            vector3D nxt_sub_cluster_packing_pos =
                vector3D(N_nexts, vector2D());

            for (int point : cur.full_sub_cluster) {
                int next_order = point_nexts_order[point];
                int next = cur.nexts[next_order];
                cur_sub_cluster_packing_pos[next_order].push_back(
                    cur.packing_pos[point]);
                nxt_sub_cluster_packing_pos[next_order].push_back(
                    packed_key_points[next].packing_pos[point]);
            }
            // cout<<"N_nexts: "<<N_nexts<<endl;
            vector1D packing_radius = vector1D(N_nexts, 0);

            for (int j = 0; j < N_nexts; j++) {
                double max_dist = 0;

                for (int k=0; k<nxt_sub_cluster_packing_pos[j].size();k++){
                    max_dist = max(max_dist, get_norm(nxt_sub_cluster_packing_pos[j][k])+radius);
                }
                packing_radius[j] = max_dist;
            }
            
            cur.nexts_packing_radius = packing_radius;
        }
    }


    // cout << "rua01\n";
    // get smooth chain with center map
    for (int i = 0; i < N_key_points; i++) {
        auto& cur = packed_key_points[i];
        int N_froms = (int)cur.froms.size();

        for (int j = 0; j < N_froms; j++) {
            int from = cur.froms[j];
            auto& pre = packed_key_points[from];
            auto edge_tuple = make_tuple(pre.id, cur.id);
            // cout << "rua01-000\n";
            vector2D smooth_chain = smooth_chain_map[edge_tuple];
            // cout << "rua01-001\n";
            int some_point = edge_point_map[edge_tuple][0];

            vector1D head_sub_cluster_center({0, 0});
            vector1D tail_sub_cluster_center({0, 0});

            if (pre.nexts.size() > 1) {
                head_sub_cluster_center = pre.next_sub_cluster_centers[some_point];
            }
            if (cur.froms.size() > 1) {
                tail_sub_cluster_center = cur.prev_sub_cluster_centers[some_point];
            }

            // cout << "rua01-002\n";

            double total_length = get_vector2D_length(smooth_chain);
            double cur_length = 0;
            int N_smooth_chain = (int)smooth_chain.size();

            vector2D smooth_chain_with_center;

            // cout << "rua01-003\n";

            for (int k = 0; k < N_smooth_chain; k++) {
                if (k > 0) {
                    cur_length += get_norm(smooth_chain[k] - smooth_chain[k - 1]);
                }

                double t = cur_length / total_length;

                vector1D cur_center = head_sub_cluster_center + (tail_sub_cluster_center - head_sub_cluster_center) * t;
                smooth_chain_with_center.push_back(cur_center + smooth_chain[k]);
            }

            smooth_chain_with_center_map[edge_tuple] = smooth_chain_with_center;
            // cout << "rua01-004\n";
        }
    }


    // cout << "rua02\n";

    // cout << "N_key_points: " << N_key_points << "\n";
    // cout << "sizeof packed_key_points: " << packed_key_points.size() << "\n";
    // cout << "    Get require cut distance\n";
    // get require cut distance
    for (int i = 0; i < N_key_points; i++) {
        auto& cur = packed_key_points[i];
        int N_nexts = (int)cur.nexts.size();
        int N_froms = (int)cur.froms.size();
        // cout<<"i = "<<i<<endl;
        // cout<<"N_nexts = "<<N_nexts<<endl;
        // cout<<"N_froms = "<<N_froms<<endl;

        // cout << "        cut on " << cur.id << endl;

        if (N_froms > 1) {
            vector3D curves(N_froms, vector2D());
            for (int j = 0; j < N_froms; j++) {
                int from = cur.froms[j];
                auto& pre = packed_key_points[from];
                auto edge_tuple = make_tuple(pre.id, cur.id);
                // if (smooth_chain_with_center_map.count(edge_tuple) == 0) {
                //     cout << "ErrorRRRR!\n";
                // }

                vector2D smooth_chain_with_center = smooth_chain_with_center_map[edge_tuple];

                if (pre.nexts.size() > 1) {
                    // smooth_chain_with_center = get_vector2D_half(smooth_chain_with_center, false);
                    smooth_chain_with_center = get_vector2D_triplet(smooth_chain_with_center, false);
                }

                curves[j] = smooth_chain_with_center;
            }
            vector1D require_dis_list = get_optimal_require_dis(curves, cur.froms_packing_radius, false);

            for (int j = 0; j < N_froms; j++) {
                int from = cur.froms[j];
                auto& pre = packed_key_points[from];
                auto edge_tuple = make_tuple(pre.id, cur.id);
                edge_tail_cut_require_map[edge_tuple] = require_dis_list[j];
            }
        }

        if (N_nexts > 1) {
            vector3D curves(N_nexts, vector2D());
            for (int j = 0; j < N_nexts; j++) {
                int next = cur.nexts[j];
                auto& nxt = packed_key_points[next];
                auto edge_tuple = make_tuple(cur.id, nxt.id);
                // if (smooth_chain_with_center_map.count(edge_tuple) == 0) {
                //     cout << "ErrorFFFF!\n";
                // }

                vector2D smooth_chain_with_center = smooth_chain_with_center_map[edge_tuple];

                if (nxt.froms.size() > 1) {
                    // smooth_chain_with_center = get_vector2D_half(smooth_chain_with_center, true);
                    smooth_chain_with_center = get_vector2D_triplet(smooth_chain_with_center, true);
                }

                curves[j] = smooth_chain_with_center;
            }
            vector1D require_dis_list = get_optimal_require_dis(curves, cur.nexts_packing_radius, true);

            for (int j = 0; j < N_nexts; j++) {
                int next = cur.nexts[j];
                auto& nxt = packed_key_points[next];
                auto edge_tuple = make_tuple(cur.id, nxt.id);
                edge_head_cut_require_map[edge_tuple] = require_dis_list[j];
            }
        }
    }

    // cout << "    Get over.\n";


    // cout << "rua03\n";
    // get edge cut
    for (int i = 0; i < N_key_points; i++) {
        auto& cur = packed_key_points[i];
        int N_froms = (int)cur.froms.size();

        for (int j = 0; j < N_froms; j++) {
            int from = cur.froms[j];
            auto& pre = packed_key_points[from];
            auto edge_tuple = make_tuple(pre.id, cur.id);
            vector2D smooth_chain_with_center = smooth_chain_with_center_map[edge_tuple];

            double head_cut_require = 0;
            double tail_cut_require = 0;

            if (pre.nexts.size() > 1) {
                head_cut_require = edge_head_cut_require_map[edge_tuple];
            }
            if (cur.froms.size() > 1) {
                tail_cut_require = edge_tail_cut_require_map[edge_tuple];
            }

            // cut!
            vector2D smooth_chain_cut = get_vector2D_cut(smooth_chain_with_center, head_cut_require, tail_cut_require);
            smooth_chain_cut_map[edge_tuple] = smooth_chain_cut;
            edge_costs[edge_tuple] = get_vector2D_length(smooth_chain_cut) / base_velocity;

            // cout << "\npre.pos : " << pre.pos[0] << " " << pre.pos[1] << endl;
            // cout << "cur.pos : " << cur.pos[0] << " " << cur.pos[1] << endl;
            // cout << "total_length : " << get_vector2D_length(smooth_chain_with_center) << endl;
            // cout << "head_cut_require : " << head_cut_require << endl;
            // cout << "tail_cut_require : " << tail_cut_require << endl;

            // cout << "smooth_chain_cut_head : " << smooth_chain_cut[0][0] << " " << smooth_chain_cut[0][1] << endl;
            // cout << "smooth_chain_cut_tail : " << smooth_chain_cut[smooth_chain_cut.size() - 1][0] << " " << smooth_chain_cut[smooth_chain_cut.size() - 1][1] << endl;
            // cout << "remain_length : " << get_vector2D_length(smooth_chain_cut) << endl;
        }
    }

    vector1D main_dir({0, 0});
    vector1D avg_start_pos({0, 0});
    vector1D avg_end_pos({0, 0});

    int N_cur_group = 0;

    for (auto& key : key_points_group) {
        if (key->DAG_inv_edges.size() == 0) {
            avg_start_pos += key->pos * (int)key->getClusterVector().size();
            N_cur_group += (int)key->getClusterVector().size();
        }
        if (key->DAG_edges.size() == 0) {
            avg_end_pos += key->pos * (int)key->getClusterVector().size();
        }
    }

    avg_start_pos = avg_start_pos / (double)N_cur_group;
    avg_end_pos = avg_end_pos / (double)N_cur_group;
    main_dir = avg_end_pos - avg_start_pos;
    main_dir = get_unit_vector(main_dir);

    auto check_is_start_key_point = [&](PackedKeyPoint& cur) {
        return cur.froms.size() == 0;
    };

    auto check_is_end_key_point = [&](PackedKeyPoint& cur) {
        return cur.nexts.size() == 0;
    };

    auto check_is_sub_cluster_start_key_point = [&](PackedKeyPoint& cur) {
        if (cur.nexts.size() == 1 && cur.froms.size() > 0) {
            int N_froms = (int)cur.froms.size();
            for (int j = 0; j < N_froms; j++) {
                int from = cur.froms[j];
                auto& pre = packed_key_points[from];
                if (!check_is_start_key_point(pre)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    };

    auto check_is_sub_cluster_end_key_point = [&](PackedKeyPoint& cur) {
        if (cur.froms.size() == 1 && cur.nexts.size() > 0) {
            int N_nexts = (int)cur.nexts.size();
            for (int j = 0; j < N_nexts; j++) {
                int next = cur.nexts[j];
                auto& nxt = packed_key_points[next];
                if (!check_is_end_key_point(nxt)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    };

    for (auto& cur : packed_key_points) {
        cur.timing = vector_dot(cur.pos, main_dir) / base_velocity;
    }

    bool have_adjusted = true;
    // bool have_adjusted = false;
    while (have_adjusted) {
        have_adjusted = false;
        for (auto& cur : packed_key_points) {
            int N_nexts = (int)cur.nexts.size();
            for (int j = 0; j < N_nexts; j++) {
                int next = cur.nexts[j];
                auto& nxt = packed_key_points[next];
                auto edge_tuple = make_tuple(cur.id, nxt.id);
                double edge_dis = get_vector2D_length(smooth_chain_with_center_map[edge_tuple]);

                bool need_to_adjust = false;
                if (nxt.timing < cur.timing) {
                    // there is a reverse edge
                    need_to_adjust = true;
                } else if (edge_dis / (nxt.timing - cur.timing) > 1.5 * base_velocity) {
                    // there is a high speed edge
                    need_to_adjust = true;
                }

                if (need_to_adjust) {
                    bool adjust_cur = false;
                    if (check_is_start_key_point(cur)) {
                        adjust_cur = true;
                    } else if (check_is_end_key_point(nxt)) {
                        adjust_cur = false;
                    } else {
                        // randomly choose one
                        adjust_cur = rand() % 2 == 0;
                    }
                    // else if (nxt.full_sub_cluster.size() > cur.full_sub_cluster.size()) {
                    //     adjust_cur = true;
                    // } else {
                    //     adjust_cur = false;
                    // }

                    if (adjust_cur) {
                        // adjust cur timing
                        // cur.timing = (cur.timing + nxt.timing - edge_dis / base_velocity) / 2;
                        cur.timing = nxt.timing - edge_dis / base_velocity;
                        // cout << "Adjust key point " << cur.id << " timing to " << cur.timing << "\n";
                        // cout << "    cur : " << cur.id << "(" << cur.pos[0] << ", " << cur.pos[1] << ")\n";
                        // cout << "    nxt : " << nxt.id << "(" << nxt.pos[0] << ", " << nxt.pos[1] << ")\n";

                    } else {
                        // adjust nxt timing
                        // nxt.timing = (nxt.timing + cur.timing + edge_dis / base_velocity) / 2;
                        nxt.timing = cur.timing + edge_dis / base_velocity;
                        // cout << "Adjust key point " << nxt.id << " timing to " << nxt.timing << "\n";
                        // cout << "    cur : " << cur.id << "(" << cur.pos[0] << ", " << cur.pos[1] << ")\n";
                        // cout << "    nxt : " << nxt.id << "(" << nxt.pos[0] << ", " << nxt.pos[1] << ")\n";
                    }
                    have_adjusted = true;
                    break;
                }
            }
            if (have_adjusted) {
                break;
            }
        }
    }

    for (auto& cur : packed_key_points) {
        if (check_is_sub_cluster_start_key_point(cur)) {
            auto& nxt = packed_key_points[cur.nexts[0]];
            auto edge_tuple = make_tuple(cur.id, nxt.id);
            double edge_dis =
                get_vector2D_length(smooth_chain_with_center_map[edge_tuple]);
            cur.timing = nxt.timing - edge_dis / (base_velocity);

        } else if (check_is_sub_cluster_end_key_point(cur)) {
            auto& pre = packed_key_points[cur.froms[0]];
            auto edge_tuple = make_tuple(pre.id, cur.id);
            double edge_dis =
                get_vector2D_length(smooth_chain_with_center_map[edge_tuple]);
            cur.timing = pre.timing + edge_dis / (base_velocity);
        }
    }

    for (auto& cur : packed_key_points) {
        if (check_is_start_key_point(cur)) {
            auto& nxt = packed_key_points[cur.nexts[0]];
            auto edge_tuple = make_tuple(cur.id, nxt.id);
            double edge_dis =
                get_vector2D_length(smooth_chain_with_center_map[edge_tuple]);
            cur.timing = nxt.timing - edge_dis / (base_velocity);

        } else if (check_is_end_key_point(cur)) {
            auto& pre = packed_key_points[cur.froms[0]];
            auto edge_tuple = make_tuple(pre.id, cur.id);
            double edge_dis =
                get_vector2D_length(smooth_chain_with_center_map[edge_tuple]);
            cur.timing = pre.timing + edge_dis / (base_velocity);
        }
    }

    map<int, double> working_times;

    for (int i = 0; i < N_key_points; i++) {
        auto& cur = packed_key_points[i];
        int N_froms = (int)cur.froms.size();
        int N_nexts = (int)cur.nexts.size();

        double cur_slow_rate = real_slow_rate[cur.id];

        working_times[cur.id] = 0;

        if (N_froms > 1) {
            vector2D start_merge_pos(N_froms, vector1D({0, 0}));
            vector2D start_merge_delays(N_froms);

            for (int j = 0; j < N_froms; j++) {
                int from = cur.froms[j];
                auto& pre = packed_key_points[from];
                auto edge_tuple = make_tuple(pre.id, cur.id);

                double required_dis = edge_tail_cut_require_map[edge_tuple];
                vector2D smooth_chain_cut = smooth_chain_cut_map[edge_tuple];
                double origin_dis = get_vector2D_length(smooth_chain_with_center_map[edge_tuple]);
                double rate = required_dis / origin_dis;

                start_merge_pos[j] = smooth_chain_cut[smooth_chain_cut.size() - 1];

                int N_components = N_components_map[edge_tuple];
                for (int k = 0; k < N_components; k++) {
                    start_merge_delays[j].push_back(rate * (cur.timing - pre.timing));
                    // start_merge_delays[j].push_back(required_dis /
                    // (trans_velocity/ cur_slow_rate ) +
                    // cur.end_merge_delays[j][k]);

                    // if (isnan(start_merge_delays[j][k])) {
                    //     cout << "Warning! start_merge_delays[" << j << "][" << k << "] is nan!" << endl;
                    //     cout << "    required_dis : " << required_dis << endl;
                    //     cout << "    key->end_merge_delays[" << j << "][" << k << "] : " << key->end_merge_delays[j][k] << endl;
                    // }
                }
            }

            cur.start_merge_delays = start_merge_delays;
            cur.start_merge_pos = start_merge_pos;

            double max_delay = numeric_limits<double>::min();
            for (int j = 0; j < N_froms; j++) {
                // max_delay = max(max_delay, start_merge_delays[j]);
                for (int k = 0; k < start_merge_delays[j].size(); k++) {
                    max_delay = max(max_delay, start_merge_delays[j][k]);
                }
            }
            working_times[cur.id] += max_delay;
            cur.merge_time = max_delay;
        }

        if (N_nexts > 1) {
            vector2D end_split_pos = vector2D(N_nexts, vector1D({0, 0}));
            // vector<double> end_split_delays = vector<double>(N_nexts, 0);
            vector2D end_split_delays = vector2D(N_nexts);

            for (int j = 0; j < N_nexts; j++) {
                int next = cur.nexts[j];
                auto& nxt = packed_key_points[next];
                auto edge_tuple = make_tuple(cur.id, nxt.id);
                double required_dis = edge_head_cut_require_map[edge_tuple];
                vector2D smooth_chain_cut = smooth_chain_cut_map[edge_tuple];
                double origin_dis = get_vector2D_length(smooth_chain_with_center_map[edge_tuple]);
                double rate = required_dis / origin_dis;

                end_split_pos[j] = smooth_chain_cut[0];

                int N_components = split_N_components_map[edge_tuple];
                for (int k = 0; k < N_components; k++) {
                    end_split_delays[j].push_back(rate * (nxt.timing - cur.timing));
                    // end_split_delays[j].push_back(required_dis / (trans_velocity/ cur_slow_rate) + cur.start_split_delays[j][k]);
                    
                // end_split_delays[j] = required_dis / (trans_velocity / cur_slow_rate) + cur.start_split_delays[j];

                // if (isnan(end_split_delays[j])) {
                //     cout << "Warning! end_split_delays[" << j << "] is nan!" << endl;
                //     cout << "    required_dis : " << required_dis << endl;
                //     cout << "    key->start_split_delays[" << j << "] : " << key->start_split_delays[j] << endl;
                // }
                }
            }

            cur.end_split_delays = end_split_delays;
            cur.end_split_pos = end_split_pos;

            double max_delay = numeric_limits<double>::min();
            for (int j = 0; j < N_nexts; j++) {
                for (int k = 0; k < end_split_delays[j].size(); k++) {
                    max_delay = max(max_delay, end_split_delays[j][k]);
                }
            }
            working_times[cur.id] += max_delay;
            cur.split_time = max_delay;
        }
    }

    for (int i = 0; i < N_key_points; i++) {
        auto& cur = packed_key_points[i];
        cur.timing_pre = cur.timing - cur.merge_time;
        cur.timing_fin = cur.timing + cur.split_time;

        // cur.timing_pre = latest_start_times[cur.id];
        // cur.timing_fin = latest_end_times[cur.id];
    }

    // compute edge speeds
    map<tuple<int, int>, double> edge_speed_map;
    for (int i = 0; i < N_key_points; i++) {
        auto& cur = packed_key_points[i];
        int N_nexts = (int)cur.nexts.size();

        for (int j = 0; j < N_nexts; j++) {
            int next = cur.nexts[j];
            auto& nxt = packed_key_points[next];
            auto edge_tuple = make_tuple(cur.id, nxt.id);

            double timing_diff = nxt.timing - cur.timing;
            double edge_length = get_vector2D_length(smooth_chain_map[edge_tuple]);

            double edge_speed = edge_length / timing_diff;
            edge_speed_map[edge_tuple] = edge_speed;
        }
    }
    
    // compute key point speeds, average of all edges
    map<int, double> key_point_speed_map;
    for (int i = 0; i < N_key_points; i++) {
        auto& cur = packed_key_points[i];
        int N_froms = (int)cur.froms.size();
        int N_nexts = (int)cur.nexts.size();

        double avg_speed = 0;

        if (N_froms == 0) {
            avg_speed = base_velocity;
        } else {
            for (int j = 0; j < N_froms; j++) {
                int from = cur.froms[j];
                auto& pre = packed_key_points[from];
                auto edge_tuple = make_tuple(pre.id, cur.id);
                avg_speed += edge_speed_map[edge_tuple];
            }
            avg_speed = avg_speed / N_froms;
        }

        key_point_speed_map[cur.id] = avg_speed;
    }

    // cout << "    Sampling Path....\n";

    for (int i = 0; i < N_key_points; i++) {
        auto& cur = packed_key_points[i];
        int N_froms = (int)cur.froms.size();
        int N_nexts = (int)cur.nexts.size();

        map<int, int> point_froms_order;
        map<int, int> point_nexts_order;

        // cout << "        Sampling on Key Point " << cur.id << endl;

        for (int j = 0; j < N_froms; j++) {
            int from = cur.froms[j];
            auto& pre = packed_key_points[from];

            auto edge_tuple = make_tuple(pre.id, cur.id);
            auto points = edge_point_map[edge_tuple];

            for (int point : points) {
                point_froms_order[point] = j;
            }
        }

        for (int j = 0; j < N_nexts; j++) {
            int next = cur.nexts[j];
            auto& nxt = packed_key_points[next];

            auto edge_tuple = make_tuple(cur.id, nxt.id);
            auto points = edge_point_map[edge_tuple];

            for (int point : points) {
                point_nexts_order[point] = j;
            }
        }

        for (int point : cur.full_sub_cluster) {
            // find the from sub-cluster
            int from_id = -1;
            int from_order = -1;
            if (N_froms > 0) {
                from_order = point_froms_order[point];
                from_id = cur.froms[from_order];
            }

            // find the next sub-cluster
            int next_id = -1;
            int next_order = -1;
            if (N_nexts > 0) {
                next_order = point_nexts_order[point];
                next_id = cur.nexts[next_order];
            }

            double time_merge_pre = cur.timing_pre;
            double time_merge_fin = cur.timing_pre + cur.merge_time;
            double time_split_pre = cur.timing_fin - cur.split_time;
            double time_split_fin = cur.timing_fin;

            if (N_froms > 1) {
                // this point has merge event
                
                auto& pre = packed_key_points[from_id];

                double time_offset_begin;

                time_offset_begin =
                    cur.merge_time - cur.start_merge_delays[from_order][cur.point_froms_cc_order[point]];

                // begin merge
                double time_begin_merge =
                    time_merge_pre + time_offset_begin;

                vector1D sub_cluster_center = {0, 0};

                // finish merge
                double time_finish_merge;

                time_finish_merge = 
                    time_merge_fin -
                    cur.end_merge_delays[from_order]
                                        [cur.point_froms_cc_order[point]];

                
                packing_pos_paths[point].push_back(PathPoint(cur.packing_pos[point], time_finish_merge, true));

                // center_paths[point].push_back(
                //     PathPoint(cur.pos, time_finish_merge, true));

                // waiting other sub-clusters
                double time_waiting_until = time_merge_fin;

                packing_pos_paths[point].push_back(PathPoint(cur.packing_pos[point], time_waiting_until, true));
                                
                // center_paths[point].push_back(
                //     PathPoint(cur.pos, time_waiting_until, true));

            }
            if (N_nexts > 1) {
                // this point has split event
            
                // waiting other sub-clusters
                auto& nxt = packed_key_points[next_id];

                double time_waiting_begin = time_split_pre;
            
                packing_pos_paths[point].push_back(PathPoint(cur.packing_pos[point], time_waiting_begin, true));

                // center_paths[point].push_back(
                //     PathPoint(cur.pos, time_waiting_begin, true));

                // begin split
                double time_begin_split;
                time_begin_split = time_split_pre + cur.start_split_delays[next_order][cur.point_nexts_cc_order[point]];

                packing_pos_paths[point].push_back(PathPoint(cur.packing_pos[point], time_begin_split, true));

                // center_paths[point].push_back(
                //     PathPoint(cur.pos, time_begin_split, true));

                // finish split
                double time_finish_split;

                time_finish_split = time_split_pre + cur.end_split_delays[next_order][cur.point_nexts_cc_order[point]];

                if (abs(nxt.merge_time) < 1e-5) {
                    // packing_pos_paths[point].push_back(
                    //     PathPoint(nxt.packing_pos[point] -
                    //                     nxt.prev_sub_cluster_centers[point],
                    //                 time_finish_split, true));
                    // packing_pos_paths[point].push_back(PathPoint(nxt.packing_pos[point], time_finish_split, true));
                } else {
                    // packing_pos_paths[point].push_back(
                    //     PathPoint(cur.packing_pos[point] -
                    //                     cur.next_sub_cluster_centers[point],
                    //                 time_finish_split, true));
                    // packing_pos_paths[point].push_back(PathPoint(cur.packing_pos[point], time_finish_split, true));
                }

                // if (get_norm(cur.pos - cur.end_split_pos[next_order]) > 1e-2) {
                //     center_paths[point].push_back(
                //         PathPoint(cur.end_split_pos[next_order],
                //                     time_finish_split, true));
                // }
                
            }
            if (N_froms == 1 && N_nexts == 1) {
                packing_pos_paths[point].push_back(
                    PathPoint(cur.packing_pos[point],
                                cur.timing_pre, true));
                // center_paths[point].push_back(
                //     PathPoint(cur.pos, cur.timing_pre, true));
            }

            if (N_froms == 0 || N_nexts == 0) {
                packing_pos_paths[point].push_back(
                    PathPoint(vector1D(2, 0), cur.timing, true));
            }

            // add sample on edge
            if (N_nexts > 0) {
                auto& nxt = packed_key_points[next_id];

                auto edge_tuple = make_tuple(cur.id, nxt.id);

                int next_from_order = -1;
                int N_nxt_froms = (int)nxt.froms.size();

                for (int j = 0; j < N_nxt_froms; j++) {
                    if (packed_key_points[nxt.froms[j]].id == cur.id) {
                        next_from_order = j;
                        break;
                    }
                }

                double head_timing = cur.timing;
                double tail_timing = nxt.timing;

                double head_velocity = key_point_speed_map[cur.id];
                double tail_velocity = key_point_speed_map[nxt.id];

                vector2D chain = smooth_chain_map[edge_tuple];

                double remain_length = get_vector2D_length(chain);

                bool use_interpolate = true;

                if (use_interpolate) {
                    // use smooth velocity path
                    auto interpolated_points = get_smooth_velocity_path(chain, head_timing, tail_timing, head_velocity, tail_velocity);

                    int N_path_points = (int)interpolated_points.size();

                    for (int j = 1; j < N_path_points - 1; j++) {
                        center_paths[point].push_back(interpolated_points[j]);
                    }
                } else {
                    // use constant velocity path
                    double cur_length = 0;

                    int N_chain = (int)chain.size();
                    for (int j = 1; j < N_chain - 1; j++) {
                    
                        vector1D center_pos = chain[j];

                        if (j > 0) {
                            cur_length += get_norm(chain[j] - chain[j - 1]);
                        }

                        double t = cur_length / remain_length;
                        double point_timing = head_timing + t * (tail_timing - head_timing);

                        center_paths[point].push_back(PathPoint(center_pos, point_timing, true));
                    }
                }
            }
        }
    
    }

    return {packing_pos_paths, center_paths};
}


vector3D sample_path(const py::object& anim,
                     const vector<vector<PathPoint>>& paths) {
    int T = anim.attr("T").cast<int>();
    int N = anim.attr("N").cast<int>();

    // find min_timing and max_timing
    auto [min_timing, max_timing] = get_min_max_timing(paths);
    // cout << "min_timing: " << min_timing << " max_timing: " << max_timing
    //      << endl;
    double cur_time = min_timing;
    double time_step = 1.0 / T;

    vector3D positions;

    while (cur_time < max_timing) {
        vector2D cur_position = get_zero2D(N, 2);
        for (int i = 0; i < N; i++) {
            cur_position[i] = interpolate_path_by_time(paths[i], cur_time);
        }
        positions.push_back(cur_position);
        cur_time += time_step;
    }

    return positions;
}

vector<vector<PathPoint>> ease_for_group(
    const py::object& anim,
    const vector<int>& group,
    const vector<vector<PathPoint>>& paths) {
    int N = (int)paths.size();
    int T = anim.attr("T").cast<int>();
    double time_step = 1.0 / T;

    string ease_function_string = anim.attr("ease_function").cast<string>();
    const char* ease_function = ease_function_string.c_str();

    double ease_ratio = anim.attr("ease_ratio").cast<double>();

    vector<vector<PathPoint>> eased_paths;
    vector<vector<PathPoint>> interpolated_paths;
    for (int i = 0; i < N; i++) {
        eased_paths.push_back(vector<PathPoint>());
        interpolated_paths.push_back(vector<PathPoint>());
    }

    for (int i : group) {
        int path_N = (int)paths[i].size();
        double start_time = paths[i][0].timing;
        double end_time = paths[i][path_N - 1].timing;
        // // get interpolated path
        // for (double t = start_time; t <= end_time; t += time_step) {
        //     vector1D pos = interpolate_path_by_time(paths[i], t);
        //     interpolated_paths[i].push_back(PathPoint(pos, t));
        // }

        // get path
        auto [ease_start, ease_end] = get_eased_start_end(
            start_time, end_time, ease_ratio, ease_function);

        for (double eased_t = ease_start; eased_t <= ease_end;
             eased_t += time_step) {
            double ori_t = apply_ease(eased_t, start_time, end_time, ease_ratio,
                                      ease_function);
            vector1D pos = interpolate_path_by_time(paths[i], ori_t);
            eased_paths[i].push_back(PathPoint(pos, eased_t));
            // get all iterations
        }
    }

    return eased_paths;
}

vector3D sample_path_combine(const py::object& anim,
                     const vector<vector<PathPoint>>& paths1,
                     const vector<vector<PathPoint>>& paths2) {
    int T = anim.attr("T").cast<int>();
    int N = anim.attr("N").cast<int>();

    // find min_timing and max_timing
    auto [min_timing1, max_timing1] = get_min_max_timing(paths1);
    auto [min_timing2, max_timing2] = get_min_max_timing(paths2);

    double min_timing = min(min_timing1, min_timing2);
    double max_timing = max(max_timing1, max_timing2);
    // cout << "min_timing: " << min_timing << " max_timing: " << max_timing
    //      << endl;
    double cur_time = min_timing;
    double time_step = 1.0 / T;

    vector3D positions;

    while (cur_time < max_timing) {
        vector2D cur_position = get_zero2D(N, 2);
        for (int i = 0; i < N; i++) {

            vector1D pos1 = interpolate_path_by_time(paths1[i], cur_time);
            // vector1D pos1(2, 0);
            vector1D pos2 = interpolate_path_by_time(paths2[i], cur_time);
            cur_position[i] = pos1 + pos2;
        }

        positions.push_back(cur_position);
        cur_time += time_step;
    }

    return positions;
}

void clean_all(vector<vector<shared_ptr<ControlPoint>>>& control_points_group,
               vector<shared_ptr<KeyPoint>>& key_points_group) {
    for (auto& control_points : control_points_group) {
        for (auto& control_point : control_points) {
            control_point->keypoint = nullptr;
        }
    }
    for (auto& key_point : key_points_group) {
        key_point->DAG_edges.clear();
        key_point->DAG_inv_edges.clear();
        key_point->points.clear();
    }
}

struct DAG_Info{
    vector1D head;
    vector1D tail;
    vector1D main_dir;
    double min_timing;
    double max_timing;
    DAG_Info(vector1D head, vector1D tail, vector1D main_dir, double min_timing, double max_timing) {
        this->head = head;
        this->tail = tail;
        this->main_dir = main_dir;
        this->min_timing = min_timing;
        this->max_timing = max_timing;
    }
    DAG_Info() {
        this->head = vector1D({0, 0});
        this->tail = vector1D({0, 0});
        this->main_dir = vector1D({0, 0});
        this->min_timing = 0;
        this->max_timing = 0;
    }
};

DAG_Info get_DAG_info(vector<vector<shared_ptr<ControlPoint>>>& control_points_group,
                    vector<shared_ptr<KeyPoint>>& key_points, 
                    double min_timing, double max_timing, 
                    const vector<int>& group) {
    vector1D main_dir({0, 0});

    for (int i : group) {
        int N_CP = (int)control_points_group[i].size();
        main_dir += control_points_group[i][N_CP - 1]->pos - control_points_group[i][0]->pos;
    }

    main_dir = get_unit_vector(main_dir);
    vector1D ortho_dir({-main_dir[1], main_dir[0]});

    double min_main_dir = numeric_limits<double>::max();
    double max_main_dir = numeric_limits<double>::min();
    double min_ortho_dir = numeric_limits<double>::max();
    double max_ortho_dir = numeric_limits<double>::min();

    for (auto& key : key_points) {
        double proj_main = vector_dot(key->pos, main_dir);
        double proj_ortho = vector_dot(key->pos, ortho_dir);
        min_main_dir = min(min_main_dir, proj_main);
        max_main_dir = max(max_main_dir, proj_main);
        min_ortho_dir = min(min_ortho_dir, proj_ortho);
        max_ortho_dir = max(max_ortho_dir, proj_ortho);
    }

    vector1D head = min_main_dir * main_dir + (min_ortho_dir + max_ortho_dir) / 2 * ortho_dir;
    vector1D tail = max_main_dir * main_dir + (min_ortho_dir + max_ortho_dir) / 2 * ortho_dir;

    return DAG_Info(head, tail, main_dir, min_timing, max_timing);
}

void get_simplified_DAG_iteratively(const py::object& anim, vector<shared_ptr<KeyPoint>>& key_points_group, map<tuple<int, int>, vector2D>& chain_map, int group_id) {
    bool operated = true;

    int iter_num = 0;

    while (operated) {
        operated = false;
        
        // cout << "    simplify...\n";
        operated |= get_simplified_DAG(key_points_group, chain_map);

        // cout << "    remove triangle...\n";
        operated |= remove_DAG_triangle_edge(key_points_group, chain_map);
        get_topo_sort_key_points(key_points_group);


        // cout << "    merge according to entropy...\n";
        operated |= get_merge_key_points_group(anim, key_points_group, chain_map);
        get_topo_sort_key_points(key_points_group);
        
        iter_num++;
    }
}

// return the animation result and the key point tree edges info
tuple<vector<vector<PathPoint>>, vector<vector<PathPoint>>, DAG_Info, double, double, double> anim_compute_for_group(
    const py::object& anim,
    const vector<int>& group,
    int group_id) {
    // use chrono to record the time
    auto start0 = chrono::high_resolution_clock::now();

    // cout << "Before get sub group" << endl;
    auto sub_groups_per_group = anim.attr("sub_groups_per_group").cast<vector<vector<vector<int>>>>();
    auto sub_groups = sub_groups_per_group[group_id];

    global_control_point_num = 0;
    global_key_point_num = 0;

    // step1 : prepare the struct of control point and key point
    // cout<<"Before get_key_points"<<endl;
    auto [control_points_group, key_points_group] = get_key_points(anim, group, sub_groups);
    auto end0 = chrono::high_resolution_clock::now();


    // step2 : build a DAG of key points
    // cout<<"Before get_DAG_of_key_points"<<endl;
    auto start1 = chrono::high_resolution_clock::now();
    get_DAG_of_key_points(anim, group, control_points_group, key_points_group);
    auto end1 = chrono::high_resolution_clock::now();

    auto start2 = chrono::high_resolution_clock::now();
    get_topo_sort_key_points(key_points_group);
    auto end2 = chrono::high_resolution_clock::now();
    double time_build_DAG = chrono::duration_cast<chrono::nanoseconds>(end0-start0+end1-start1+end2-start2).count()  / 1000000000.0;

    // step3 : get the smooth curve of DAG
    // cout<<"Before get_smooth_DAG"<<endl;
    auto chain_map = get_smooth_chain_map(anim, group, key_points_group);

    // step4 : simplify the DAG
    get_simplified_DAG_iteratively(anim, key_points_group, chain_map, group_id);

    // step5 : pack the key points
    // cout<<"Before get_packing_of_key_points"<<endl;
    auto start5 = chrono::high_resolution_clock::now();
    auto packed_key_points = get_packing_of_key_points(anim, key_points_group, control_points_group, group_id);
    auto end5 = chrono::high_resolution_clock::now();
    double time_packing = chrono::duration_cast<chrono::nanoseconds>(end5-start5).count() / 1000000000.0;

    // early return
    if (anim.attr("quit_after_packing").cast<bool>()) {
        int N = anim.attr("N").cast<int>();
        return {vector<vector<PathPoint>>(N), vector<vector<PathPoint>>(N), DAG_Info(), time_build_DAG, time_packing, 0};
    }

    // step6 : generate paths according to the packed key points
    // cout<<"Before get_packing_path"<<endl;
    auto start6 = chrono::high_resolution_clock::now();
    auto [packing_paths, center_paths] = get_packing_path(
        anim, group, control_points_group, key_points_group, packed_key_points, chain_map, group_id);

    packing_paths = ease_for_group(anim, group, packing_paths);
    center_paths = ease_for_group(anim, group, center_paths);
    
    auto end6 = chrono::high_resolution_clock::now();
    double time_packing_path = chrono::duration_cast<chrono::nanoseconds>(end6-start6).count() / 1000000000.0;

    auto [min_timing1, max_timing1] = get_min_max_timing(packing_paths);
    auto [min_timing2, max_timing2] = get_min_max_timing(center_paths);
    double min_timing = min(min_timing1, min_timing2);
    double max_timing = max(max_timing1, max_timing2);
    // cout<<"min_timing: "<<min_timing<<" max_timing: "<<max_timing<<endl;
    auto dag_info = get_DAG_info(control_points_group, key_points_group, min_timing, max_timing, group);

    // double time_build_DAG = chrono::duration_cast<chrono::nanoseconds>(end0-start0+end1-start1+end2-start2+end3-start3+end4-start4+end5-start5+end6-start6).count() / 1000000000.0;

    // step7 : clean the memory
    // cout << "Before clean" << endl;
    clean_all(control_points_group, key_points_group);
    

    return {packing_paths, center_paths, dag_info, time_build_DAG, time_packing, time_packing_path};
}

vector<int> get_DAGs_order(const py::object& anim, const vector<DAG_Info> dag_infos) {
    int k = anim.attr("amount_groups").cast<int>();
    double DAG_merge_eps = anim.attr("DAG_merge_eps").cast<double>();
    double DAG_merge_checkdot = anim.attr("DAG_merge_checkdot").cast<double>();

    vector<vector<int>> edges(k);
    vector<int> deg(k, 0);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (i == j) {
                continue;
            }
            if (vector_dot(dag_infos[i].main_dir, dag_infos[j].main_dir) <= DAG_merge_checkdot) {
                continue;
            }
            if (get_norm(dag_infos[i].tail - dag_infos[j].head) <= DAG_merge_eps) {
                // i -> j
                edges[i].push_back(j);
                deg[j]++;

                // cout << "Find DAG can merge : " << i << " -> " << j << endl;
            }
        }
    }

    // only for test
    vector<int> order;
    
    queue<int> q;
    for (int i = 0; i < k; i++) {
        if (!deg[i]) {
            q.push(i);
        }
    }

    while(!q.empty()) {
        int cur = q.front();
        q.pop();
        order.push_back(cur);
        for (int next : edges[cur]) {
            deg[next]--;
            if (!deg[next]) {
                q.push(next);
            }
        }
    }

    return order;
}

tuple<vector3D,
      vector3D,
      vector<int>,
      vector<int>>
anim_compute(const py::object& anim) {
    // get start and end position from animation
    vector2D start_position = anim.attr("start_position").cast<vector2D>();
    vector2D end_position = anim.attr("end_position").cast<vector2D>();
    vector<vector<int>> anime_groups =
        anim.attr("anime_groups").cast<vector<vector<int>>>();

    int N = anim.attr("N").cast<int>();
    int k = anim.attr("amount_groups").cast<int>();
    int T = anim.attr("T").cast<int>();

    vector<vector<PathPoint>> all_packing_points(N);
    vector<vector<PathPoint>> all_bundle_points(N);

    vector<int> anime_start(N);
    vector<int> anime_end(N);

    vector4D edges_info_for_group;

    double timing_dag = 0;
    double timing_packing = 0;
    double timing_path = 0;

    vector<DAG_Info> dag_infos;

    for (int clusteridx = 0; clusteridx < k; clusteridx++) {
        // cout<<"clusteridx: "<<clusteridx<<endl;
        vector<int> group;
        // cout<<"anime_groups.size(): "<<anime_groups.size()<<endl;
        // cout<<"anime_groups[0].size(): "<<anime_groups[0].size()<<endl;
        for (int i = 0; i < N; i++) {
            // cout<<"i: "<<i<<" clusteridx: "<<clusteridx<<" anime_groups[i][clusteridx]: "<<anime_groups[i][clusteridx]<<endl;
            if (anime_groups[i][clusteridx] == 1) {
                group.push_back(i);
            }
        }

        // get the animation of current group
        auto [packing_path, bundled_positions_path, dag_info, time_build_DAG, time_packing, time_packing_path] =
            anim_compute_for_group(anim, group, clusteridx);
        
        dag_infos.push_back(dag_info);
        
        for (int i : group) {
            all_packing_points[i] = packing_path[i];
            all_bundle_points[i] = bundled_positions_path[i];
        }

        timing_dag += time_build_DAG;
        timing_packing += time_packing;
        timing_path += time_packing_path;
        // cout<<"timing_dag: "<<timing_dag<<" timing_packing: "<<timing_packing<<" timing_path: "<<timing_path<<endl;


        // for (int i : group) {
        //     anime_end[i] = (int)positions.size() - 1;
        // }
    }

    // cout << "    DAG timing : " << timing_dag << " sec" << endl;
    // cout << "    Packing timing : " << timing_packing << " sec" << endl;

    if (anim.attr("quit_after_packing").cast<bool>()) {
        return {vector3D(), vector3D(), vector<int>(), vector<int>()};
    }

    // sort the DAGs
    double cur_time_offset = 0;
    auto order = get_DAGs_order(anim, dag_infos);
    for (int cluster_idx : order) {
        vector<int> group;
        for (int i = 0; i < N; i++) {
            if (anime_groups[i][cluster_idx] == 1) {
                group.push_back(i);
            }
        }

        for (int i : group) {
            int N_path1 = (int)all_packing_points[i].size();
            for (int j = 0; j < N_path1; j++) {
                all_packing_points[i][j].timing += cur_time_offset - dag_infos[cluster_idx].min_timing;
            }

            int N_path2 = (int)all_bundle_points[i].size();
            for (int j = 0; j < N_path2; j++) {
                all_bundle_points[i][j].timing += cur_time_offset - dag_infos[cluster_idx].min_timing;
            }
        }

        cur_time_offset += dag_infos[cluster_idx].max_timing - dag_infos[cluster_idx].min_timing;

        // add 5 frame between each cluster
        cur_time_offset += 5.0 / T;
    }

    // sample all the path
    // vector3D positions = sample_path(anim, all_packing_points);
    vector3D positions = sample_path_combine(anim, all_packing_points, all_bundle_points);
    vector3D bundled_positions = sample_path(anim, all_bundle_points);

    // cout<<"timing_dag: "<<timing_dag<<" timing_packing: "<<timing_packing<<" timing_path: "<<timing_path<<endl;
    // detect end
    int T_all = (int)positions.size();
    for (int i = 0; i < N; i++) {
        for (int t = T_all - 1; t > 0; t--) {
            vector1D pos_0 = positions[t - 1][i];
            vector1D pos_1 = positions[t][i];

            if (get_norm(pos_0 - pos_1) > 1e-5) {
                anime_end[i] = t;
                break;
            }
        }

        for (int t = 0; t < T_all - 1; t++) {
            vector1D pos_0 = positions[t][i];
            vector1D pos_1 = positions[t + 1][i];

            if (get_norm(pos_0 - pos_1) > 1e-5) {
                anime_start[i] = t;
                break;
            }
        }
    }

    return {positions, bundled_positions, anime_start, anime_end};
}

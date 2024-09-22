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

#include "pdutils.h"
#include "vectorND.h"
#define M_PI 3.14159265358979323846

using namespace std;
namespace py = pybind11;

tuple<double, double, double> find_circle(vector2D& positions){
    vector1D center(2);
    double radius = 0;
    min_circle_cover(positions, center, radius);
    return make_tuple(center[0], center[1], radius);
    // double max_dist = -1;
    // pair<int, int> max_pair;
    // for (int i = 0; i < positions.size(); i++){
    //     for (int j = i+1; j < positions.size(); j++){
    //         double dist = (positions[i][0]-positions[j][0])* (positions[i][0]-positions[j][0]) + (positions[i][1]-positions[j][1])*(positions[i][1]-positions[j][1]);
    //         if (dist > max_dist){
    //             max_dist = dist;
    //             max_pair = make_pair(i, j);
    //         }
    //     }
    // }
    // max_dist = -1;
    // int p1 = max_pair.first;
    // int p2 = max_pair.second;
    // int p3 = -1;
    // for (int i = 0; i < positions.size(); i++){
    //     if (i == p1 || i == p2){
    //         continue;
    //     }
    //     double dist = abs((positions[p2][1] - positions[p1][1]) * positions[i][0]
    //                  - (positions[p2][0] - positions[p1][0]) * positions[i][1]
    //                  + positions[p2][0] * positions[p1][1] - positions[p2][1] * positions[p1][0])
    //                   / sqrt((positions[p2][1] - positions[p1][1]) * (positions[p2][1] - positions[p1][1]) + (positions[p2][0] - positions[p1][0]) * (positions[p2][0] - positions[p1][0]));
    //     if (dist > max_dist){
    //         max_dist = dist;
    //         p3 = i;
    //     }
    // }
    // double x1 = positions[p1][0];
    // double y1 = positions[p1][1];
    // double x2 = positions[p2][0];
    // double y2 = positions[p2][1];
    // double x3 = positions[p3][0];
    // double y3 = positions[p3][1];
    // double D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
    // double center_x = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / D;
    // double center_y = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / D;
    // double radius = sqrt((x1 - center_x) * (x1 - center_x) + (y1 - center_y) * (y1 - center_y));
    // return make_tuple(center_x, center_y, radius);
}

void circle_center3(vector1D p0, vector1D p1, vector1D p2, vector1D& cp) {
    double a = p1[0] - p0[0];
    double b = p1[1] - p0[1];
    double c = p2[0] - p0[0];
    double d = p2[1] - p0[1];
    double e = a * (p0[0] + p1[0]) + b * (p0[1] + p1[1]);
    double f = c * (p0[0] + p2[0]) + d * (p0[1] + p2[1]);
    double g = 2 * (a * (p2[1] - p1[1]) - b * (p2[0] - p1[0]));
    if (g == 0) {
        cp[0] = 0;
        cp[1] = 0;
    } else {
        cp[0] = (d * e - b * f) / g;
        cp[1] = (a * f - c * e) / g;
    }
}

bool point_in(vector1D p, vector1D c, double r) {
    return (p[0] - c[0]) * (p[0] - c[0]) + (p[1] - c[1]) * (p[1] - c[1]) <= r * r;
}

void min_circle_cover(vector2D pos, vector1D& center, double& radius){
    radius = 0;
    center = pos[0];
    for (int i = 1; i < pos.size(); i++){
        if (!point_in(pos[i], center, radius)){
            center = pos[i];
            radius = 0;
            for (int j = 1; j < i; j++){
                if (!point_in(pos[j], center, radius)){
                    center[0] = (pos[i][0] + pos[j][0]) / 2;
                    center[1] = (pos[i][1] + pos[j][1]) / 2;
                    radius = 0.5 * sqrt((pos[i][0] - pos[j][0]) * (pos[i][0] - pos[j][0]) + (pos[i][1] - pos[j][1]) * (pos[i][1] - pos[j][1]));
                    for (int k = 1; k < j; k++){
                        if (!point_in(pos[k], center, radius)){
                            circle_center3(pos[i], pos[j], pos[k], center);
                            radius = sqrt((pos[i][0] - center[0]) * (pos[i][0] - center[0]) + (pos[i][1] - center[1]) * (pos[i][1] - center[1]));
                        }
                    }
                }
            }
        }
    }
}



vector2D interpolate(double center_x, double center_y, double radius, int pcs){
    vector2D result;
    for (int i = 0; i < pcs; i++){
        double angle = i * 2 * M_PI / pcs;
        double x = center_x + radius * cos(angle);
        double y = center_y + radius * sin(angle);
        result.push_back({x, y});
    }
    return result;
}

vector2D multi_interpolate(const vector2D& centers, const vector1D& radii){
    int times = 5;
    vector2D result;
    for (int i = 0; i < centers.size(); i++){
        vector2D circle = interpolate(centers[i][0], centers[i][1], radii[i]*1.05, times);
        result.insert(result.end(), circle.begin(), circle.end());
    }
    return result;
}

tuple<double,double> cell_centroid(const vector2D& cell){
    double x = 0;
    double y = 0;
    double area = 0;
    for (int i = 0; i < cell.size(); i++){
        double x1 = cell[i][0];
        double y1 = cell[i][1];
        double x2 = cell[(i+1)%cell.size()][0];
        double y2 = cell[(i+1)%cell.size()][1];
        double temp = x1*y2 - x2*y1;
        x += (x1 + x2) * temp;
        y += (y1 + y2) * temp;
        area += temp;
    }
    area *= 3.0;
    return make_tuple(x/area, y/area);
}

double inscribed_circle_radius(const vector2D& cell, const vector1D& site){
    double r = 1e5;
    for (int i = 0; i < cell.size(); i++){
        double x1 = cell[i][0];
        double y1 = cell[i][1];
        double x2 = cell[(i+1)%cell.size()][0];
        double y2 = cell[(i+1)%cell.size()][1];
        double edge_length = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
        if (edge_length < 1e-10){
            continue;
        }
        // temp = cross_product(p1-site, p2-site)
        double temp = (x1-site[0])*(y2-site[1]) - (x2-site[0])*(y1-site[1]);
        r = min(r, abs(temp)/edge_length);
    }
    return r;
}
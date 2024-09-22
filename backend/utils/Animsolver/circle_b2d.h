#include <fstream>
#include <iostream>
#include <ctime>
#include <vector>
#include <cmath>
#include "box2d/box2d.h"
#include "vectorND.h"


class Circle_B2D
{
public: 
    int n;
    double* positions;
    double* radii;
    int n_attraction_pairs;
    int* attraction_pairs;
    double gravity_mag;
    double attraction_mag;
    double size_mag;
    double alpha;
    double alpha_min;
    double alpha_decay;
    

    b2World* world;
    std::vector<b2Body*> circles;
    std::vector<b2Body*> hulls;
    void initBodies(int n, double* positions, double* radii, double size_mag);
    void initForces(int n_attraction_pairs, int* attraction_pairs, double gravity_mag, double attraction_mag);
    void calculateForces(double* forces);
};

vector2D Simulate(int n, double* positions, double* radii, int n_attraction_pairs, int* attractions, double size_mag, double gravity_mag, double attraction_mag, int n_iters, double alpha_min);

vector2D LeaveContourSpcae(int n, double* positions, double* radii, double size_mag, std::vector<std::vector<int>> sub_clusters);
void SaveImage(int n, double* positions, double* radii, std::vector<std::vector<int>> edges, vector2D hulls=vector2D(0));
vector2D GlobalPacking(int n, double* positions, double* radii, std::vector<std::vector<int>> sub_clusters, vector2D hulls, double size_mag);
vector2D GlobalPackingBubbleTree(
    const vector2D& init_pos,
    const vector1D& radii,
    const std::vector<std::vector<int>>& sub_clusters,
    double contour_width);

void SaveImageHull(vector3D hulls);

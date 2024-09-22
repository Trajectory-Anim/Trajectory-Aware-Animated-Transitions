#include "circle_b2d.h"
#include "opencv2/opencv.hpp"
#include <omp.h>
#include <chrono>
#define PI acos(-1)

void Circle_B2D::initBodies(int n, double* positions, double* radii, double size_mag)
{
    this->n = n;
    this->positions = positions;
    this->radii = radii;
    for (int i=0; i<n; i++){
        radii[i] *= size_mag;
        positions[2*i] *= size_mag;
        positions[2*i+1] *= size_mag;
    }
    
    this->circles = std::vector<b2Body*>(n);
    for (int i = 0; i < n; i++)
    {
        b2BodyDef bodyDef;
        bodyDef.type = b2_dynamicBody;
        bodyDef.position.Set(positions[2 * i], positions[2 * i + 1]);
        bodyDef.linearDamping = 0.0f;
        bodyDef.angularDamping = 0.0f;
        bodyDef.angle = 0.0f;
        bodyDef.bullet = false;
        b2Body* body = world->CreateBody(&bodyDef);
        b2CircleShape circle;
        circle.m_radius = radii[i];
        b2FixtureDef fixtureDef;
        fixtureDef.shape = &circle;
        fixtureDef.density = 0.0f;
        fixtureDef.friction = 0.0f;
        body->CreateFixture(&fixtureDef);
        this->circles[i] = body;
    }
    // for (int i = 0; i < n; i++){
    //     std::cout<<"position: "<<this->circles[i]->GetPosition().x<<", "<<this->circles[i]->GetPosition().y<<std::endl;
    // }
}

void Circle_B2D::initForces(int n_attraction_pairs, int* attraction_pairs, double gravity_mag, double attraction_mag)
{
    this->n_attraction_pairs = n_attraction_pairs;
    this->attraction_pairs = attraction_pairs;
    this->gravity_mag = gravity_mag;
    this->attraction_mag = attraction_mag;
}


void Circle_B2D::calculateForces(double *forces){
    double* centroid = new double[2];
    centroid[0] = 0;
    centroid[1] = 0;
    for (int i=0;i<this->n;i++){
        // printf("i: %d\n", i);
        this->positions[2*i] = this->circles[i]->GetPosition().x;
        this->positions[2*i+1] = this->circles[i]->GetPosition().y;
        // printf("position %d: %f, %f\n", i, this->positions[2 * i], this->positions[2 * i + 1]);
        centroid[0] += this->positions[2*i];
        centroid[1] += this->positions[2*i+1];
    }
    centroid[0] /= this->n;
    centroid[1] /= this->n;
    // printf("centroid: %f, %f\n", centroid[0], centroid[1]);
    for (int i=0;i<this->n;i++){
        double dx = this->positions[2*i] - centroid[0];
        double dy = this->positions[2*i+1] - centroid[1];
        double dist = sqrt(dx*dx + dy*dy);
        if (dist == 0){ // avoid division by zero
            continue;
        }
        forces[2*i] = -gravity_mag * dx / dist * this->alpha*this->size_mag;
        forces[2*i+1] = -gravity_mag * dy / dist * this->alpha*this->size_mag;
    }
    if (this->n_attraction_pairs == 0){
        delete[] centroid;
        return;
    }
    // get attractions
    for (int i=0;i<this->n_attraction_pairs;i++){
        int i1 = (int)attraction_pairs[2*i];
        int i2 = (int)attraction_pairs[2*i+1];
        double dx = this->positions[2*i1] - this->positions[2*i2];
        double dy = this->positions[2*i1+1] - this->positions[2*i2+1];
        double dist = sqrt(dx*dx + dy*dy);
        if (dist == 0){ // avoid division by zero
            continue;
        }
        forces[2*i1] -= attraction_mag * dx / dist * this->alpha *this->size_mag;
        forces[2*i1+1] -= attraction_mag * dy / dist * this->alpha*this->size_mag;
        forces[2*i2] += attraction_mag * dx / dist * this->alpha*this->size_mag;
        forces[2*i2+1] += attraction_mag * dy / dist * this->alpha*this->size_mag;
    }
    // output forces
    // for (int i=0;i<this->n;i++){
    //     printf("force %d: %f, %f\n", i, forces[2*i], forces[2*i+1]);
    // }
    delete[] centroid;
}

vector2D Simulate(int n, double* positions, double* radii, int n_attraction_pairs, int* attractions, double size_mag, double gravity_mag, double attraction_mag, int n_iters, double alpha_min){
    Circle_B2D circle_b2d;
    circle_b2d.world = new b2World(b2Vec2(0, 0));
    circle_b2d.initBodies(n, positions, radii, size_mag);
    circle_b2d.n_attraction_pairs = n_attraction_pairs;
    circle_b2d.attraction_pairs = attractions;
    circle_b2d.size_mag = size_mag;
    circle_b2d.gravity_mag = gravity_mag;
    circle_b2d.attraction_mag = attraction_mag;
    circle_b2d.alpha_min = alpha_min;
    circle_b2d.alpha_decay = (1-(pow(circle_b2d.alpha_min, 1.0/n_iters)));
    circle_b2d.alpha = 1;
    double* forces = new double[2 * n];
    vector2D out_positions(n, vector1D(2));
    // set forces to zero
    memset(forces, 0, 2 * n * sizeof(double));
    

    for (int i = 0; i < n_iters; i++)
    {
        // compare time for calculate forces and step

        // get time
        // auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < n; j++)
        {
            circle_b2d.circles[j]->SetLinearVelocity(b2Vec2(0, 0));
            circle_b2d.circles[j]->SetAngularVelocity(0);
        }
        circle_b2d.calculateForces(forces);
        // get positions before step

        for (int j = 0; j < n; j++)
        {
            circle_b2d.circles[j]->ApplyForceToCenter(b2Vec2(forces[2 * j], forces[2 * j + 1]), true);
        }
        circle_b2d.world->Step(0.005, 6, 2);
        circle_b2d.alpha += circle_b2d.alpha_decay * (0 - circle_b2d.alpha);

    }
    for (int i = 0; i < n; i++)
    {
        out_positions[i][0] = circle_b2d.circles[i]->GetPosition().x/size_mag;
        out_positions[i][1] = circle_b2d.circles[i]->GetPosition().y/size_mag;
    }
    delete[] forces;
    delete circle_b2d.world;
    double* draw_positions = new double[2 * n];
    for (int i = 0; i < n; i++)
    {
        draw_positions[2 * i] = out_positions[i][0];
        draw_positions[2 * i + 1] = out_positions[i][1];
        radii[i] /= size_mag;
    }
    std::vector<std::vector<int>> edges;
    for (int i = 0; i < n_attraction_pairs; i++)
    {
        std::vector<int> edge;
        edge.push_back(attractions[2 * i]);
        edge.push_back(attractions[2 * i + 1]);
        edges.push_back(edge);
    }
    // SaveImage(n, draw_positions, radii, edges);
    delete[] draw_positions;
    return out_positions;
}

void SaveImage(int n, double* positions, double* radii, std::vector<std::vector<int>> edges, vector2D hulls){
    // save image
    int width = 1024;
    int height = 1024;
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    double min_x = 1e9;
    double max_x = -1e9;
    double min_y = 1e9;
    double max_y = -1e9;
    for (int i = 0; i < n; i++)
    {
        min_x = std::min(min_x, positions[2 * i] - radii[i]);
        max_x = std::max(max_x, positions[2 * i] + radii[i]);
        min_y = std::min(min_y, positions[2 * i + 1] - radii[i]);
        max_y = std::max(max_y, positions[2 * i + 1] + radii[i]);
    }
    double scale = std::max(max_x - min_x, max_y - min_y);

    double* draw_positions = new double[2 * n];
    for (int i = 0; i < n; i++)
    {
        draw_positions[2 * i] = (positions[2 * i] - min_x) / scale * width;
        draw_positions[2 * i + 1] = (positions[2 * i + 1] - min_y) / scale * height;
    }
    double* draw_radii = new double[n];
    for (int i = 0; i < n; i++)
    {
        draw_radii[i] = radii[i] / scale * width;
    }
    for (int i = 0; i < n; i++)
    {
        cv::circle(img, cv::Point(draw_positions[2 * i], draw_positions[2 * i + 1]), draw_radii[i], cv::Scalar(255, 255, 255), 2);
        cv::putText(img, std::to_string(i), cv::Point(draw_positions[2 * i], draw_positions[2 * i + 1]), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }
    for (int i = 0; i < edges.size(); i++)
    {
        cv::line(img, cv::Point(draw_positions[2 * edges[i][0]], draw_positions[2 * edges[i][0] + 1]), cv::Point(draw_positions[2 * edges[i][1]], draw_positions[2 * edges[i][1] + 1]), cv::Scalar(255, 255, 255), 2);
    }
    if (hulls.size()>0){
        // vector3D draw_hulls = vector3D();
        // for (int i = 0; i < hulls.size(); i++){
        //     vector2D draw_hull = vector2D();
        //     for (int j = 0; j < hulls[i].size(); j++){
        //         vector1D draw_point = vector1D();
        //         draw_point.push_back((hulls[i][j][0] - min_x) / scale * width);
        //         draw_point.push_back((hulls[i][j][1] - min_y) / scale * height);
        //         draw_hull.push_back(draw_point);
        //     }
        //     draw_hulls.push_back(draw_hull);
        // }
        // for (int i = 0; i < draw_hulls.size(); i++){
        //     for (int j = 0; j < draw_hulls[i].size(); j++){
        //         cv::line(img, cv::Point(draw_hulls[i][j][0], draw_hulls[i][j][1]), cv::Point(draw_hulls[i][(j+1)%draw_hulls[i].size()][0], draw_hulls[i][(j+1)%draw_hulls[i].size()][1]), cv::Scalar(255, 255, 255), 2);
        //     }
        // }
        vector2D draw_hulls = vector2D();
        for (int i = 0; i < hulls.size(); i++){
            vector1D draw_hull = vector1D();
            draw_hull.push_back((hulls[i][0] - min_x) / scale * width);
            draw_hull.push_back((hulls[i][1] - min_y) / scale * height);
            draw_hull.push_back(hulls[i][2] / scale * width);
            draw_hulls.push_back(draw_hull);
        }
        for (int i = 0; i < draw_hulls.size(); i++){
            cv::circle(img, cv::Point(draw_hulls[i][0], draw_hulls[i][1]), draw_hulls[i][2], cv::Scalar(255, 255, 255), 2);
        }
    }
    // get a time stamp in nanoseconds
    auto now = std::chrono::system_clock::now();
    auto now_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
    // convert this time stamp to string
    std::string now_ns_str = std::to_string(now_ns.time_since_epoch().count());
    std::string filename = "./log/image/PackingProcess/" + now_ns_str + ".png";
    cv::imwrite(filename, img);

    delete[] draw_positions;
}

void SaveImageHull(vector3D hulls){
    // save image
    int width = 1024;
    int height = 1024;
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    vector2D hull_points = vector2D();
    for (int i = 0; i < hulls.size(); i++){
        for (int j = 0; j < hulls[i].size(); j++){
            hull_points.push_back(hulls[i][j]);
            std::cout<<"draw hull point: "<<hulls[i][j][0]<<", "<<hulls[i][j][1]<<std::endl;
        }
    }

    double min_x = 1e9;
    double max_x = -1e9;
    double min_y = 1e9;
    double max_y = -1e9;
    for (int i = 0; i < hull_points.size(); i++)
    {
        min_x = std::min(min_x, hull_points[i][0]);
        max_x = std::max(max_x, hull_points[i][0]);
        min_y = std::min(min_y, hull_points[i][1]);
        max_y = std::max(max_y, hull_points[i][1]);
    }
    double scale = std::max(max_x - min_x, max_y - min_y);
    std::cout<<"min_x: "<<min_x<<" max_x: "<<max_x<<" min_y: "<<min_y<<" max_y: "<<max_y<<std::endl;
    std::cout<<"scale: "<<scale<<std::endl;
    vector3D draw_hulls = vector3D();
    // cout<<"hull size: "<<hulls.size()<<endl;
    for (int i = 0; i < hulls.size(); i++){
        // cout<<"hull "<<i<<endl;
        vector2D draw_hull = vector2D();
        for (int j = 0; j < hulls[i].size(); j++){
            vector1D draw_point = vector1D();
            draw_point.push_back((hulls[i][j][0] - min_x) / scale * width);
            draw_point.push_back((hulls[i][j][1] - min_y) / scale * height);
            cout<<"draw_point: "<<draw_point[0]<<", "<<draw_point[1]<<endl;
            draw_hull.push_back(draw_point);
        }
        draw_hulls.push_back(draw_hull);
    }
    for (int i = 0; i < draw_hulls.size(); i++){
        for (int j = 0; j < draw_hulls[i].size(); j++){
            cv::line(img, cv::Point(draw_hulls[i][j][0], draw_hulls[i][j][1]), cv::Point(draw_hulls[i][(j+1)%draw_hulls[i].size()][0], draw_hulls[i][(j+1)%draw_hulls[i].size()][1]), cv::Scalar(255, 255, 255), 2);
        }
    }
    // get a time stamp in nanoseconds
    auto now = std::chrono::system_clock::now();
    auto now_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
    // convert this time stamp to string
    std::string now_ns_str = std::to_string(now_ns.time_since_epoch().count());
    std::string filename = "./log/image/PackingProcess/" + now_ns_str + ".png";
    cv::imwrite(filename, img);
}

void SaveImageContour(int n,
               double* positions,
               double* radii,
               double contour_width,
               int iter) {
    // save image
    int width = 1024;
    int height = 1024;
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    double min_x = 1e9;
    double max_x = -1e9;
    double min_y = 1e9;
    double max_y = -1e9;
    for (int i = 0; i < n; i++) {
        min_x = std::min(min_x, positions[2 * i] - radii[i]);
        max_x = std::max(max_x, positions[2 * i] + radii[i]);
        min_y = std::min(min_y, positions[2 * i + 1] - radii[i]);
        max_y = std::max(max_y, positions[2 * i + 1] + radii[i]);
    }
    double scale = std::max(max_x - min_x, max_y - min_y);

    double* draw_positions = new double[2 * n];
    for (int i = 0; i < n; i++) {
        draw_positions[2 * i] = (positions[2 * i] - min_x) / scale * width;
        draw_positions[2 * i + 1] =
            (positions[2 * i + 1] - min_y) / scale * height;
    }
    double* draw_radii = new double[n];
    double* draw_radii_contour = new double[n];
    for (int i = 0; i < n; i++) {
        draw_radii[i] = radii[i] / scale * width;
        draw_radii_contour[i] = (radii[i] + contour_width) / scale * width;
    }
    for (int i = 0; i < n; i++) {
        cv::circle(img,
                   cv::Point(draw_positions[2 * i], draw_positions[2 * i + 1]),
                   draw_radii[i], cv::Scalar(255, 255, 255), 2);
        cv::circle(img,
                   cv::Point(draw_positions[2 * i], draw_positions[2 * i + 1]),
                   draw_radii_contour[i], cv::Scalar(100, 100, 100), 2);

        cv::putText(img, std::to_string(i),
                    cv::Point(draw_positions[2 * i], draw_positions[2 * i + 1]),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255),
                    2);
    }

    // get a time stamp in nanoseconds
    auto now = std::chrono::system_clock::now();
    auto now_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
    // convert this time stamp to string
    std::string now_ns_str = std::to_string(now_ns.time_since_epoch().count());
    std::string iter_name = std::to_string(iter);
    std::string filename = "./log/image/PackingProcess/globalpack_" + now_ns_str + "_iter_" + iter_name +".png";
    cv::imwrite(filename, img);

    delete[] draw_radii;
    delete[] draw_radii_contour;
    delete[] draw_positions;
}

vector2D LeaveContourSpcae(int n, double* positions, double* radii, double size_mag, std::vector<std::vector<int>> sub_clusters){
    Circle_B2D circle_b2d;
    circle_b2d.world = new b2World(b2Vec2(0, 0));
    double enlarge = 1.2;
    double* enlarged_radii = new double[n];
    for (int i = 0; i < n; i++)
    {
        enlarged_radii[i] = radii[i] * enlarge;
    }
    circle_b2d.initBodies(n, positions, enlarged_radii, size_mag);
    circle_b2d.size_mag = size_mag;
    circle_b2d.gravity_mag = 0.0;
    circle_b2d.attraction_mag = 0;
    circle_b2d.n_attraction_pairs = 0;
    circle_b2d.alpha = 1;
    double* forces = new double[2 * n];
    vector2D out_positions(n, vector1D(2));
    int n_iters = 5;
    // set forces to zero
    memset(forces, 0, 2 * n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        circle_b2d.circles[i]->SetLinearVelocity(b2Vec2(0, 0));
        circle_b2d.circles[i]->SetAngularVelocity(0);
    }
    std::vector<std::vector<int>> edges = std::vector<std::vector<int>>();
    // create distance joints to fix the distance between the circles
    for (int i = 0; i < sub_clusters.size(); i++)
    {
        for (int j = 0; j < sub_clusters[i].size(); j++)
        {
            for (int k = j + 1; k < sub_clusters[i].size(); k++)
            {
                b2DistanceJointDef jointDef;
                jointDef.bodyA = circle_b2d.circles[sub_clusters[i][j]];
                jointDef.bodyB = circle_b2d.circles[sub_clusters[i][k]];
                jointDef.localAnchorA = b2Vec2(0, 0);
                jointDef.localAnchorB = b2Vec2(0, 0);
                double distance = b2Distance(circle_b2d.circles[sub_clusters[i][j]]->GetPosition(), circle_b2d.circles[sub_clusters[i][k]]->GetPosition());
                jointDef.length = distance;
                jointDef.stiffness = 100000000;
                jointDef.collideConnected = false;
                circle_b2d.world->CreateJoint(&jointDef);
                // std::cout<<"joint created between "<<sub_clusters[i][j]<<" and "<<sub_clusters[i][k]<<std::endl;
                // std::cout<<"length: "<<jointDef.length<<std::endl;
                // std::cout<<"position1: "<<positions[2 * sub_clusters[i][j]]<<", "<<positions[2 * sub_clusters[i][j] + 1]<<std::endl;
                // std::cout<<"position2: "<<positions[2 * sub_clusters[i][k]]<<", "<<positions[2 * sub_clusters[i][k] + 1]<<std::endl;
                std::vector<int> edge;
                edge.push_back(sub_clusters[i][j]);
                edge.push_back(sub_clusters[i][k]);
                edges.push_back(edge);
            }
        }
    }
    for (int i = 0; i < n_iters; i++)
    {
        circle_b2d.calculateForces(forces);
        for (int j = 0; j < n; j++)
        {
            circle_b2d.circles[j]->ApplyForceToCenter(b2Vec2(forces[2 * j], forces[2 * j + 1]), true);
        }
        circle_b2d.world->Step(0.005, 6, 2);
    }
    for (int i = 0; i < n; i++)
    {
        out_positions[i][0] = circle_b2d.circles[i]->GetPosition().x / size_mag;
        out_positions[i][1] = circle_b2d.circles[i]->GetPosition().y / size_mag;
    }
    double* draw_radii = new double[n];
    // for (int i = 0; i < n; i++)
    // {
    //     enlarged_radii[i] /= enlarge;
    // }
    // SaveImage(n, positions, enlarged_radii, edges);
    delete[] forces;
    delete circle_b2d.world;
    return out_positions;
}


vector2D GlobalPacking(int n, double* positions, double* radii, std::vector<std::vector<int>> sub_clusters, vector2D hulls, double size_mag){
    Circle_B2D circle_b2d;
    circle_b2d.world = new b2World(b2Vec2(0, 0));
    int valid_n = 0;
    for (auto cluster : sub_clusters){
        valid_n += cluster.size();
    }
    map<int, int> idx_map;
    map<int, int> idx_map_rev;
    double* valid_positions = new double[2 * valid_n];
    double* valid_radii = new double[valid_n];
    int cur_idx = 0;
    for (auto sub_cluster : sub_clusters){
        for (auto idx : sub_cluster){
            idx_map[idx] = cur_idx;
            idx_map_rev[cur_idx] = idx;
            valid_positions[2 * cur_idx] = positions[2 * idx];
            valid_positions[2 * cur_idx + 1] = positions[2 * idx + 1];
            valid_radii[cur_idx] = radii[idx];
            cur_idx++;
        }
    }
    // std::cout<<"valid_n: "<<valid_n<<std::endl;
    vector2D scaled_hulls = vector2D();
    for (int i = 0; i < hulls.size(); i++){
        vector1D scaled_hull = vector1D();
        // for (int j = 0; j < hulls[i].size(); j++){
        //     vector1D scaled_point = vector1D();
        //     scaled_point.push_back(hulls[i][j][0] * size_mag);
        //     scaled_point.push_back(hulls[i][j][1] * size_mag);
        //     scaled_hull.push_back(scaled_point);
        //     if (j == 0){
        //         std::cout<<"hull point: "<<scaled_point[0]<<", "<<scaled_point[1]<<std::endl;
        //     }
        // }
        for (int j = 0; j < hulls[i].size(); j++){
            scaled_hull.push_back(hulls[i][j]*size_mag);
        }
        scaled_hulls.push_back(scaled_hull);
    }
    circle_b2d.initBodies(valid_n, valid_positions, valid_radii, size_mag);
    vector2D start_positions = vector2D();
    for (int i=0; i<valid_n; i++){
        start_positions.push_back({valid_positions[2*i], valid_positions[2*i+1]});
    }
    circle_b2d.size_mag = size_mag;
    circle_b2d.alpha = 1;
    int n_clusters = sub_clusters.size();
    circle_b2d.gravity_mag = 50.0;
    vector2D start_hull_positions = vector2D();
    for (int i = 0; i < sub_clusters.size(); i++){
        // std::cout<<"i: "<<i<<std::endl;
        std::vector<int> cluster = sub_clusters[i];
        vector1D hull = scaled_hulls[i];

        // // convert hull to b2Vec2
        // b2Vec2* hull_b2d = new b2Vec2[hull.size()];
        // for (int j = 0; j < hull.size(); j++){
        //     hull_b2d[j].Set(hull[j][0], hull[j][1]);
        // }
        // for (int j = 0; j < hull.size(); j++){
        //     std::cout<<"hull point: "<<hull_b2d[j].x<<", "<<hull_b2d[j].y<<std::endl;
        // }
        // // init shapes
        // b2PolygonShape hull_shape;
        // // hull_shape.Set((b2Vec2*)hull.data(), hull.size());
        // hull_shape.Set(hull_b2d, hull.size());
        // std::cout<<"hull size: "<<hull.size()<<endl;
        // b2Vec2 center = hull_shape.m_centroid;
        // std::cout<<"vertices: "<<hull_shape.m_count<<std::endl;
        // std::cout<<"approximate radius: "<<hull_shape.m_radius<<std::endl;
        // std::cout<<"center: "<<center.x<<", "<<center.y<<std::endl;
        // start_positions.push_back({center.x, center.y});

        // // init bodies
        // b2BodyDef hull_body;
        // hull_body.type = b2_dynamicBody;
        // hull_body.position.Set(0, 0);
        // b2Body* body = circle_b2d.world->CreateBody(&hull_body);
        // body->CreateFixture(&hull_shape, 0.0f);
        // circle_b2d.hulls.push_back(body);
        // std::cout<<"hull created"<<endl;

        b2BodyDef bodyDef;
        bodyDef.type = b2_dynamicBody;
        bodyDef.position.Set(hull[0],hull[1]);
        start_hull_positions.push_back({hull[0], hull[1]});
        bodyDef.linearDamping = 0.0f;
        bodyDef.angularDamping = 0.0f;
        bodyDef.angle = 0.0f;
        bodyDef.bullet = false;
        b2Body* body = circle_b2d.world->CreateBody(&bodyDef);
        b2CircleShape circle;
        circle.m_radius = hull[2];
        b2FixtureDef fixtureDef;
        fixtureDef.shape = &circle;
        fixtureDef.density = 0.0f;
        fixtureDef.friction = 0.0f;
        body->CreateFixture(&fixtureDef);
        circle_b2d.hulls.push_back(body);

        for (int idx = 0; idx < cluster.size(); idx++){
            // std::cout<<"idx: "<<idx<<std::endl;
            b2WeldJointDef jointDef;
            jointDef.bodyA = circle_b2d.hulls[i];
            // std::cout<<"hull position: "<<circle_b2d.hulls[i]->GetPosition().x<<", "<<circle_b2d.hulls[i]->GetPosition().y<<std::endl;
            jointDef.bodyB = circle_b2d.circles[idx_map[cluster[idx]]];
            // std::cout<<"circle_idx: "<<idx_map[cluster[idx]]<<std::endl;
            // std::cout<<"circle position: "<<circle_b2d.circles[cluster[idx_map[idx]]]->GetPosition().x<<", "<<circle_b2d.circles[cluster[idx_map[idx]]]->GetPosition().y<<std::endl;
            jointDef.localAnchorA = b2Vec2(0, 0);
            jointDef.localAnchorB = b2Vec2(0, 0);
            jointDef.collideConnected = false;
            circle_b2d.world->CreateJoint(&jointDef);
        }
    }
    // std::cout<<"init done"<<std::endl;
    int n_iters = 100;
    double* forces = new double[2 * n_clusters];
    for (int iter = 0; iter < n_iters; iter++)
    {
        // cout<<"iter: "<<iter<<endl;
        // vector3D draw_hulls = vector3D();
        vector2D draw_hulls = vector2D();
        for (int i=0;i<sub_clusters.size();i++){
            // std::cout<<"hull i: "<<i<<std::endl;
            // vector2D now_hull = vector2D();
            vector1D now_hull = vector1D();
            auto hull_body = circle_b2d.hulls[i];
            b2Vec2 body_position = hull_body->GetPosition();
            // double body_angle = hull_body->GetAngle();
            // std::cout<<"position: "<<body_position.x<<", "<<body_position.y<<std::endl;
            now_hull.push_back(body_position.x);
            now_hull.push_back(body_position.y);
            now_hull.push_back(hull_body->GetFixtureList()->GetShape()->m_radius);
            // std::cout<<"angle: "<<body_angle<<std::endl;
            // auto hull_shape = (b2PolygonShape*)hull_body->GetFixtureList()->GetShape();
            // for (int j=0;j<hull_shape->m_count;j++){
            //     // vector1D now_point = vector1D();
            //     // now_point.push_back(hull_body->GetPosition().x + hull_shape->m_vertices[j].x);
            //     // now_point.push_back(hull_body->GetPosition().y + hull_shape->m_vertices[j].y);
            //     // now_hull.push_back(now_point);
            //     b2Vec2 point = hull_shape->m_vertices[j];
            //     b2Rot rot(body_angle);
            //     b2Vec2 rotated_point = b2Mul(rot, point);
            //     std::cout<<"rotate point: "<<rotated_point.x<<", "<<rotated_point.y<<std::endl;
            //     b2Vec2 world_point = body_position + rotated_point;
            //     vector1D now_point = vector1D();
            //     now_point.push_back(world_point.x);
            //     now_point.push_back(world_point.y);
            //     now_hull.push_back(now_point);
            //     std::cout<<"world point: "<<world_point.x<<", "<<world_point.y<<std::endl;
            // }
            draw_hulls.push_back(now_hull);
            // vector1D now_center = vector1D();
            // now_center.push_back(circle_b2d.hulls[i]->GetPosition().x);
            // now_center.push_back(circle_b2d.hulls[i]->GetPosition().y);
            // for (int j=0;j<hulls[i].size();j++){
            //     vector1D now_point = vector1D();
            //     now_point.push_back(circle_b2d.hulls[i]->GetPosition().x + scaled_hulls[i][j][0]);
            //     now_point.push_back(circle_b2d.hulls[i]->GetPosition().y + scaled_hulls[i][j][1]);
            //     now_hull.push_back(now_point);
            // }
            // draw_hulls.push_back(now_hull);
            
        }

        // update circles positions
        for (int i = 0 ; i < sub_clusters.size(); i++){
            // std::cout<<"cluster i: "<<i<<std::endl;
            std::vector<int> cluster = sub_clusters[i];
            b2Vec2 cur_center = circle_b2d.hulls[i]->GetPosition();
            b2Vec2 start_center = b2Vec2(start_hull_positions[i][0], start_hull_positions[i][1]);
            b2Vec2 move = cur_center - start_center;
            // std::cout<<"center: "<<cur_center.x<<", "<<cur_center.y<<std::endl;
            // std::cout<<"start center: "<<start_center.x<<", "<<start_center.y<<std::endl;
            // std::cout<<"move: "<<move.x<<", "<<move.y<<std::endl;
            for (int j = 0; j < cluster.size(); j++){
                b2Vec2 start_position = b2Vec2(start_positions[idx_map[cluster[j]]][0], start_positions[idx_map[cluster[j]]][1]);
                b2Vec2 new_position = start_position + move;
                // std::cout<<"circle j: "<<idx_map[cluster[j]]<<std::endl;
                // std::cout<<"start position: "<<start_position.x<<", "<<start_position.y<<std::endl;
                // std::cout<<"new position: "<<new_position.x<<", "<<new_position.y<<std::endl;
                // set the position of circle as the new position
                circle_b2d.circles[idx_map[cluster[j]]]->SetTransform(new_position, 0);
            }
        }
        // SaveImageHull(draw_hulls);

        double* centroid = new double[2];
        centroid[0] = 0;
        centroid[1] = 0;
        for (int i=0;i<circle_b2d.n;i++){
            // printf("i: %d\n", i);
            circle_b2d.positions[2*i] = circle_b2d.circles[i]->GetPosition().x;
            circle_b2d.positions[2*i+1] = circle_b2d.circles[i]->GetPosition().y;
            // printf("position %d: %f, %f\n", i, this->positions[2 * i], this->positions[2 * i + 1]);
            centroid[0] += circle_b2d.positions[2*i];
            centroid[1] += circle_b2d.positions[2*i+1];
        }
        centroid[0] /= circle_b2d.n;
        centroid[1] /= circle_b2d.n;
        // printf("centroid: %f, %f\n", centroid[0], centroid[1]);
        for (int i=0;i<n_clusters;i++){
            double dx = circle_b2d.hulls[i]->GetPosition().x - centroid[0];
            double dy = circle_b2d.hulls[i]->GetPosition().y - centroid[1];
            double dist = sqrt(dx*dx + dy*dy);
            if (dist == 0){ // avoid division by zero
                continue;
            }
            // forces[2*i] = -circle_b2d.gravity_mag * dx / dist * circle_b2d.alpha*((double(n_iters-iter))/(double(n_iters)))*circle_b2d.size_mag;
            // forces[2*i+1] = -circle_b2d.gravity_mag * dy / dist * circle_b2d.alpha*((double(n_iters-iter))/(double(n_iters)))*circle_b2d.size_mag;
            forces[2*i] = -circle_b2d.gravity_mag * dx / dist * circle_b2d.alpha*circle_b2d.size_mag;
            forces[2*i+1] = -circle_b2d.gravity_mag * dy / dist * circle_b2d.alpha*circle_b2d.size_mag;
            // std::cout<<"force: "<<forces[2*i]<<", "<<forces[2*i+1]<<std::endl;
            circle_b2d.hulls[i]->ApplyForceToCenter(b2Vec2(forces[2*i], forces[2*i+1]), true);
        }
        if (iter % 20 == 0){
            // SaveImage(circle_b2d.n, circle_b2d.positions, circle_b2d.radii, std::vector<std::vector<int>>(), draw_hulls);
        }
        circle_b2d.world->Step(0.005, 6, 2);
        for (int i=0;i<n_clusters;i++){
            // check if the hull's velocity
            // b2Vec2 velocity = circle_b2d.hulls[i]->GetLinearVelocity();
            // std::cout<<"velocity: "<<velocity.x<<", "<<velocity.y<<std::endl;
            // clear the velocity
            circle_b2d.hulls[i]->SetLinearVelocity(b2Vec2(0, 0));
        }
    }
    vector2D out_positions(n, vector1D(2, 0));
    for (int i = 0; i < n; i++)
    {
        out_positions[i][0] = positions[2 * i];
        out_positions[i][1] = positions[2 * i + 1];
    }
    for (int i = 0; i < valid_n; i++){
        out_positions[idx_map_rev[i]][0] = circle_b2d.circles[i]->GetPosition().x / size_mag;
        out_positions[idx_map_rev[i]][1] = circle_b2d.circles[i]->GetPosition().y / size_mag;
    }
    delete[] valid_positions;
    delete[] valid_radii;
    delete[] forces;
    delete circle_b2d.world;
    return out_positions;
}

vector2D GlobalPackingBubbleTree(
    const vector2D& init_pos,
    const vector1D& radii,
    const std::vector<std::vector<int>>& sub_clusters,
    double contour_width) {
    // cout << "Init pos : " << init_pos.size() << ", " << endl;
    // cout << "Radii : " << radii.size() << ", " << endl;
    // cout << "Sub clusters : ";
    // for (int i = 0; i < sub_clusters.size(); i++){
    //     cout << "    sub cluster " << i << " : ";
    //     for (int j = 0; j < sub_clusters[i].size(); j++){
    //         cout << sub_clusters[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    auto start_time = chrono::high_resolution_clock::now();

    double size_mag = 200;

    vector2D sub_clusters_centers(sub_clusters.size(), vector1D(2, 0));
    for (int i = 0; i < sub_clusters.size(); i++){
        for (int j : sub_clusters[i]) {
            sub_clusters_centers[i] += init_pos[j] * size_mag;
        }
        sub_clusters_centers[i] /= sub_clusters[i].size();
    }

    // cout << "rua1\n";
    b2World* world = new b2World(b2Vec2(0, 0));

    // init bodies, each body is a cluster
    vector<b2Body*> bodies;
    for (int i = 0; i < sub_clusters.size(); i++) {
        double rand_x = sub_clusters_centers[i][0] + 1e-5 * rand() / (double)RAND_MAX;
        double rand_y = sub_clusters_centers[i][1] + 1e-5 * rand() / (double)RAND_MAX;

        b2BodyDef bodyDef;
        bodyDef.type = b2_dynamicBody;
        bodyDef.position.Set(rand_x, rand_y);
        b2Body* body = world->CreateBody(&bodyDef);

        for (int j : sub_clusters[i]) {
            b2CircleShape circle;
            vector1D local_pos = init_pos[j] * size_mag - sub_clusters_centers[i];
            circle.m_p.Set(local_pos[0], local_pos[1]);
            circle.m_radius = (radii[j] + contour_width) * size_mag;
            b2FixtureDef fixtureDef;
            fixtureDef.shape = &circle;
            fixtureDef.density = 0.0f;
            fixtureDef.friction = 0.0f;
            body->CreateFixture(&fixtureDef);
        }
        
        bodies.push_back(body);
    }

    // cout << "rua2\n";

    // compute centroid
    vector1D centroid(2, 0);
    int num_of_circle = 0;
    for (int i = 0; i < sub_clusters.size(); i++) {
        centroid += sub_clusters_centers[i] * (int)sub_clusters[i].size();
        num_of_circle += (int)sub_clusters[i].size();
    }
    centroid /= num_of_circle;

    // cout << "rua3\n";

    auto end_time = chrono::high_resolution_clock::now();

    double time_prepare = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / 1000000000.0;

    start_time = chrono::high_resolution_clock::now();

    // compute forces
    // int n_iters = 500;
    int n_iters = 100;
    double gravity_mag = 1e3;
    for (int iter = 0; iter < n_iters; iter++) {
        for (int i = 0; i < sub_clusters.size(); i++) {
            vector1D cur_pos = vector1D({bodies[i]->GetPosition().x, bodies[i]->GetPosition().y});
            double dis = get_norm(centroid - cur_pos);

            if (dis < 1e-5) {
                continue;
            }

            vector1D force = get_unit_vector(centroid - cur_pos) * dis * dis * gravity_mag;
            bodies[i]->ApplyForceToCenter(b2Vec2(force[0], force[1]), true);
        }
        world->Step(0.005, 6, 2);

        // clear the velocity
        for (int i = 0; i < sub_clusters.size(); i++) {
            bodies[i]->SetLinearVelocity(b2Vec2(0, 0));
        }

    }

    end_time = chrono::high_resolution_clock::now();

    double time_iter = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / 1000000000.0;

    // cout << "rua4\n";

    vector2D out_positions(init_pos.size(), vector1D(2, 0));

    for (int i = 0; i < sub_clusters.size(); i++) {
        vector1D final_center = vector1D({bodies[i]->GetPosition().x, bodies[i]->GetPosition().y});
        for (int j : sub_clusters[i]) {
            out_positions[j] = (init_pos[j] * size_mag - sub_clusters_centers[i] + final_center) / size_mag;
        }
    }

    // cout << "        time_gb_prepare : " << time_prepare << endl;
    // cout << "        time_gb_iter : " << time_iter << endl;

    // cout << "rua5\n";

    delete world;

    return out_positions;
}

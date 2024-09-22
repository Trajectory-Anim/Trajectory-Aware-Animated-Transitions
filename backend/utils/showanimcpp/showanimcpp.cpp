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

#include "opencv2/opencv.hpp"
#include "showanimcpp.h"
#include "vectorND.h"
#define THREADS_NUM 48

using namespace std;
using namespace cv;
namespace py = pybind11;

Scalar get_color(int index, bool isTransparent = false, double alpha = 0.1) {
    index = index % 10; // Ensure index is within the range of 0 to 9
    // Define the RGB values for the tab10 colormap
    vector<tuple<int, int, int>> tab10_colors = {
        make_tuple(31, 119, 180), // 0
        make_tuple(255, 127, 14), // 1
        make_tuple(44, 160, 44), // 2
        make_tuple(214, 39, 40), // 3
        make_tuple(148, 103, 189), // 4
        make_tuple(140, 86, 75), // 5
        make_tuple(227, 119, 194), // 6
        make_tuple(127, 127, 127), // 7
        make_tuple(188, 189, 34), // 8
        make_tuple(23, 190, 207) // 9
    };

    tuple<int, int, int> color = make_tuple(200, 200, 200);
    if (index >= 0) {
        color = tab10_colors[index];
    }

    if (isTransparent) {
        // Apply transparency effect
        color = make_tuple(
            min(255, static_cast<int>(alpha * get<0>(color) + (1 - alpha) * 255)),
            min(255, static_cast<int>(alpha * get<1>(color) + (1 - alpha) * 255)),
            min(255, static_cast<int>(alpha * get<2>(color) + (1 - alpha) * 255))
        );
    }
    // Return the color in BGR format
    return Scalar(get<2>(color), get<1>(color), get<0>(color));
}

Scalar get_color_gradient(tuple<int, int, int> origin_color, double ratio) {
    int r = static_cast<int>((1 - ratio) * get<0>(origin_color));
    int g = static_cast<int>((1 - ratio) * get<1>(origin_color));
    int b = static_cast<int>((1 - ratio) * get<2>(origin_color));
    return Scalar(r, g, b);
}

void show_animation(vector3D anime_position,
                    vector<vector<int>> cluster,
                    vector<int> anime_start,
                    vector<int> anime_end,
                    vector1D radii,
                    int width,
                    int height,
                    string title,
                    int fps,
                    double margin_percentage,
                    double fade_time,
                    string save_to_filename,
                    int resolution_scale,
                    vector3D bundled_position,
                    bool trace_mode,
                    vector<int> points_to_trace,
                    int grid_n) {

    width *= resolution_scale;
    height *= resolution_scale;

    if (bundled_position.size() == 0) {
        bundled_position = anime_position;
    }

    int N = (int) anime_position[0].size();
    int amount_frames = (int) anime_position.size();
    int fade_frames = (int) (fade_time * fps);
    int N_cluster = (int) cluster[0].size();
    vector<int> start_of_cluster(N_cluster, numeric_limits<int>::max());
    vector<int> end_of_cluster(N_cluster, numeric_limits<int>::min());

    for (int clusteridx = 0; clusteridx < N_cluster; clusteridx++) {
        for (int i = 0; i < N; i++) {
            if (cluster[i][clusteridx] == 1) {
                start_of_cluster[clusteridx] = min(start_of_cluster[clusteridx], anime_start[i]);
                end_of_cluster[clusteridx] = max(end_of_cluster[clusteridx], anime_end[i]);
            }
        }
    }

    vector<int> cluster_order(N_cluster, 0);
    for (int i = 0; i < N_cluster; i++) {
        cluster_order[i] = i;

        // cout << "start : " << start_of_cluster[i] << ", end : " << end_of_cluster[i] << "\n";
    }

    sort(cluster_order.begin(), cluster_order.end(), [&](int a, int b) {
        return start_of_cluster[a] > start_of_cluster[b];
    });

    vector<int> frame_to_copy;
    for (int start : start_of_cluster) {
        frame_to_copy.push_back(start);
    }
    for (int end : end_of_cluster) {
        frame_to_copy.push_back(end);
    }
    sort(frame_to_copy.begin(), frame_to_copy.end());
    frame_to_copy.erase(unique(frame_to_copy.begin(), frame_to_copy.end()), frame_to_copy.end());

    // random trace point
    vector<bool> is_trace_point(N, false);
    // int trace_num = 3;
    // int trace_num = 1;
    // for (int i = 0; i < trace_num; i++) {
    //     int idx = rand() % N;
    //     while (is_trace_point[idx]) {
    //         idx = rand() % N;
    //     }
    //     is_trace_point[idx] = true;
    // }
    // is_trace_point[1] = true;

    for (int point : points_to_trace) {
        is_trace_point[point] = true;
    }


    for (int frame_id : frame_to_copy) {
        if (frame_id >= amount_frames) {
            cout << "Warning! frame " << frame_id << " is out of range!" << endl;
            return;
        }


        vector2D cur_anime_pos = anime_position[frame_id];
        vector2D cur_bundled_pos = bundled_position[frame_id];

        // insert the frame into anime_position and bundled_position
        for (int i = 0; i < fade_frames; i++) {
            anime_position.insert(anime_position.begin() + frame_id, cur_anime_pos);
            bundled_position.insert(bundled_position.begin() + frame_id, cur_bundled_pos);
        }

        // affect the frame id behind
        amount_frames += fade_frames;
        for (int i = 0; i < N; i++) {
            if (anime_start[i] > frame_id) {
                anime_start[i] += fade_frames;
            }
            if (anime_end[i] > frame_id) {
                anime_end[i] += fade_frames;
            }
        }

        for (int i = 0; i < N_cluster; i++) {
            if (start_of_cluster[i] > frame_id) {
                start_of_cluster[i] += fade_frames;
            }
            if (end_of_cluster[i] > frame_id) {
                end_of_cluster[i] += fade_frames;
            }
        }
        
        int len_frame_to_copy = (int) frame_to_copy.size();
        for (int i = 0; i < len_frame_to_copy; i++) {
            if (frame_to_copy[i] > frame_id) {
                frame_to_copy[i] += fade_frames;
            }
        }

    }

    // extra wait for start
    if (trace_mode) {
        int copy_num = fade_frames * 2;
        for (int i = 0; i < copy_num; i++) {
            anime_position.insert(anime_position.begin(), anime_position[0]);
            bundled_position.insert(bundled_position.begin(), bundled_position[0]);
        }

        amount_frames += copy_num;
        for (int i = 0; i < N; i++) {
            anime_start[i] += copy_num;
            anime_end[i] += copy_num;
        }

        for (int i = 0; i < N_cluster; i++) {
            start_of_cluster[i] += copy_num;
            end_of_cluster[i] += copy_num;
        }
    }

    auto get_image_x = [&](double origin_x) {
        origin_x = origin_x * (1 - 2 * margin_percentage) + margin_percentage;
        return (int) (origin_x * width);
    };

    auto get_image_y = [&](double origin_y) {
        origin_y = origin_y * (1 - 2 * margin_percentage) + margin_percentage;
        origin_y = 1 - origin_y;
        return (int) (origin_y * height);
    };

    auto get_single_image = [&](int timepoint) {
        vector2D positions = anime_position[timepoint];
        // Mat image = Mat::zeros(height, width, CV_8UC3);
        Mat image(height, width, CV_8UC3, Scalar(255, 255, 255));

        // draw grid
        if (grid_n >= 2) {
            // draw x grid
            auto color = Scalar(240, 240, 240);

            for (int i = 1; i < grid_n; i++) {
                double t = (double)i / (double)grid_n;
                int x = t * width;
                line(image, Point(x, 0), Point(x, height - 1), color, 4 * resolution_scale, LINE_AA);
            }
            line(image, Point(0, 0), Point(0, height - 1), color, 4 * resolution_scale, LINE_AA);
            line(image, Point(width - 1, 0), Point(width - 1, height - 1), color, 4 * resolution_scale, LINE_AA);

            // draw y grid
            for (int i = 1; i < grid_n; i++) {
                double t = (double)i / (double)grid_n;
                int y = t * height;
                line(image, Point(0, y), Point(width - 1, y), color, 4 * resolution_scale, LINE_AA);
            }
            line(image, Point(0, 0), Point(width - 1, 0), color, 4 * resolution_scale, LINE_AA);
            line(image, Point(0, height - 1), Point(width - 1, height - 1), color, 4 * resolution_scale, LINE_AA);
        }

        vector<int> draw_order(N, 0);
        for (int i = 0; i < N; i++) {
            draw_order[i] = i;
        }

        auto order_func = [&](int a) {
            if (timepoint > anime_end[a]) {
                return 100000 + anime_end[a];
            } else if (timepoint < anime_start[a]) {
                return 100000 + anime_start[a];
            } else {
                return anime_end[a];
            }
        };

        sort(draw_order.begin(), draw_order.end(), [&](int a, int b) {
            return order_func(a) > order_func(b);
        });

        vector<int> clusteridx(N, 0);
        vector<int> circle_size(N, 0);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N_cluster; j++) {
                if (cluster[i][j] == 1) {
                    clusteridx[i] = j;
                    break;
                }
            }
            circle_size[i] = (int)(radii[i] * width) - 1;
        }

        for (int i : draw_order) {
            int x = get_image_x(positions[i][0]);
            int y = get_image_y(positions[i][1]);
            Scalar color;

            if (trace_mode) {
                color = Scalar(180, 180, 180);
            } else {
                color = get_color(clusteridx[i], true);
            }

            if (timepoint >= anime_end[i] + fade_frames) {
                circle(image, Point(x, y), circle_size[i], color, -1, LINE_AA);
            }
        }

        if (!trace_mode){
            // draw the trace of already passed
            for (int i : draw_order) {
                if (timepoint > start_of_cluster[clusteridx[i]] && timepoint < end_of_cluster[clusteridx[i]] + fade_frames) {
                    for (int t = anime_start[i]; t < anime_end[i] + fade_frames; t++) {
                        if (timepoint < t + 2) {
                            continue;
                        }

                        int tx1 = get_image_x(bundled_position[t][i][0]);
                        int ty1 = get_image_y(bundled_position[t][i][1]);
                        int tx2 = get_image_x(bundled_position[t + 1][i][0]);
                        int ty2 = get_image_y(bundled_position[t + 1][i][1]);

                        double fade_ratio = 0;

                        if (timepoint >= end_of_cluster[clusteridx[i]]) {
                            fade_ratio = 1 - (double)(timepoint - end_of_cluster[clusteridx[i]]) / fade_frames;
                        } else {
                            fade_ratio = 1;
                        }

                        Scalar color = get_color(-1, true, fade_ratio);

                        line(image, Point(tx1, ty1), Point(tx2, ty2), color, circle_size[i] * 2 + 1, LINE_AA);
                    }
                }
            }
        }

        // for (int i : draw_order) {
        //     if (timepoint > start_of_cluster[clusteridx[i]]) {
        //         for (int t = anime_start[i]; t < anime_end[i]; t++) {
        //             Scalar color(200, 200, 200);
        //             if (timepoint >= t) {
        //                 continue;
        //             }

        //             int tx1 = get_image_x(bundled_position[t][i][0]);
        //             int ty1 = get_image_y(bundled_position[t][i][1]);
        //             int tx2 = get_image_x(bundled_position[t + 1][i][0]);
        //             int ty2 = get_image_y(bundled_position[t + 1][i][1]);

        //             line(image, Point(tx1, ty1), Point(tx2, ty2), color, circle_size[i] * 2 + 1, LINE_AA);
        //         }
        //     }
        // }

        for (int i : draw_order) {
            int x = get_image_x(positions[i][0]);
            int y = get_image_y(positions[i][1]);
            
            if (timepoint >= anime_end[i] + fade_frames) {
                continue;
            }

            int fade_in_start = start_of_cluster[clusteridx[i]];
            int fade_out_start = anime_end[i];

            Scalar color;

            if (trace_mode) {
                // fade_in_start = anime_start[i];
                fade_in_start = start_of_cluster[clusteridx[i]];

                if (timepoint < fade_in_start) {
                    if (is_trace_point[i]) {
                        color = Scalar(0, 0, 255);
                    } else {
                        color = Scalar(180, 180, 180);
                    }
                } else if (timepoint < fade_in_start + fade_frames) {

                    double black_ratio = (double)(timepoint - fade_in_start) / fade_frames;
                    if (is_trace_point[i]) {
                        color = get_color_gradient(make_tuple(0, 0, 255), black_ratio);
                    } else {
                        color = get_color_gradient(make_tuple(180, 180, 180), black_ratio);
                    }

                } else if (timepoint >= fade_out_start && timepoint < fade_out_start + fade_frames) {
                    
                    double black_ratio = 1 - (double)(timepoint - fade_out_start) / fade_frames;
                    color = get_color_gradient(make_tuple(180, 180, 180), black_ratio);

                } else {
                    color = Scalar(0, 0, 0);
                }
            } else {
                double fade_ratio = 0;

                if (timepoint < start_of_cluster[clusteridx[i]]) {
                    continue;
                } else if (timepoint >= fade_in_start && timepoint <= fade_in_start + fade_frames) {
                    fade_ratio = (double)(timepoint - fade_in_start) / fade_frames;
                } else if (timepoint >= fade_out_start && timepoint <= fade_out_start + fade_frames) {
                    fade_ratio = 1 - (double)(timepoint - fade_out_start) / fade_frames;
                } else {
                    fade_ratio = 1;
                }

                color = get_color(clusteridx[i], true, fade_ratio);
            }

            circle(image, Point(x, y), circle_size[i], color, -1, LINE_AA);
        }

        putText(image, title, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, get_color(0), 1, LINE_AA);

        return image;
    };

    string filepath = "./log/video/" + save_to_filename;

    VideoWriter videoWriter(filepath, VideoWriter::fourcc('m','p','4','v'), fps, Size(width, height));
    if (!videoWriter.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return;
    }

    vector<Mat> frames(amount_frames);

    auto time_start = chrono::steady_clock::now();

    int count = 0;
#pragma omp parallel for
    for (int i = 0; i < amount_frames; ++i) {
        frames[i] = get_single_image(i);

        #pragma omp critical
        {
            count++;
            if (count % 10 == 0) {
                cout << "Calculating frame " << count << "/" << amount_frames << "\r";
            }
        }
    }

    auto time_end = chrono::steady_clock::now();
    double time_elapsed = chrono::duration_cast<chrono::duration<double>>(time_end - time_start).count();
    // cout << "Computing Animation Frame Time: " << time_elapsed << " seconds" << endl;

    for (int i = 0; i < amount_frames; ++i) {
        videoWriter.write(frames[i]);
    }

    videoWriter.release();
    cout << "Saved to " << filepath << endl;
    return;
}

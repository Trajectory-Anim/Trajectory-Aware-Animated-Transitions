import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from shapely.geometry import LineString
from tqdm import tqdm
import pickle
from utils.interpolate.interpolate import *

PATH = './animation/dataset/'

def get_squeeze_path(path, target_start, target_end, squeeze_rate=0.2):
    l = len(path)

    total_length = 0
    for i in range(l - 1):
        total_length += np.linalg.norm(np.array(path[i]) - np.array(path[i+1]))

    cur_length = 0
    new_path = []
    for i in range(l):
        t = cur_length / total_length
        shift = (target_start + (target_end - target_start) * t - np.array(path[i]))
        cur_squeeze_rate = squeeze_rate + (1 - squeeze_rate) * np.abs(t - 0.5) * 2

        new_pos = np.array(path[i]) + shift * cur_squeeze_rate

        new_path.append(new_pos.tolist())

        if i < l - 1:
            cur_length += np.linalg.norm(np.array(path[i+ 1]) - np.array(path[i]))
    
    return new_path

def convert_np_to_list(data):
    if type(data) == np.ndarray:
        result = data.tolist()
    elif type(data) == list:
        result = []
        for item in data:
            result.append(convert_np_to_list(item))
    else:
        result = data
    
    return result

def save_data(filename, data):
    with open(PATH + filename + '.json', 'w') as file:
        json.dump(data, file, indent=4)

def load_data(filename):
    with open(PATH + filename + '.json', 'r') as file:
        data = json.load(file)
    return data

def load_data_weight(filename):
    with open(PATH + filename + '_sizes.json', 'r') as file:
        data = json.load(file)
    return data

def normalize_data(data):
    # normalized data to [0, 1]
    N = len(data)
    x_min, x_max = 1e5, -1e5
    y_min, y_max = 1e5, -1e5
    for i in range(N):
        pathN = len(data[i])
        for j in range(pathN):
            x_min = min(x_min, data[i][j][0])
            x_max = max(x_max, data[i][j][0])
            y_min = min(y_min, data[i][j][1])
            y_max = max(y_max, data[i][j][1])
    
    scale = max(x_max - x_min, y_max - y_min)

    normed_data = []
    for i in range(N):
        # if i % 100==0:
        #     print(f'Processing {i}/{N}')
        pathN = len(data[i])
        normed_path = []
        for j in range(pathN):
            normed_path.append([
                # (data[i][j][0] - x_min) / scale,
                # (data[i][j][1] - y_min) / scale
                (data[i][j][0] + (scale - x_max - x_min) / 2) / scale,
                (data[i][j][1] + (scale - y_max - y_min) / 2) / scale
            ])
        normed_data.append(normed_path)

    return normed_data


def normalize_data2(data):
    # normalized data to [0, 1]
    N = len(data)
    x_min, x_max = 1e5, -1e5
    y_min, y_max = 1e5, -1e5
    for i in range(N):
        pathN = len(data[i])
        for j in range(pathN):
            x_min = min(x_min, data[i][j][0])
            x_max = max(x_max, data[i][j][0])
            y_min = min(y_min, data[i][j][1])
            y_max = max(y_max, data[i][j][1])

    scale = max(x_max - x_min, y_max - y_min)

    normed_data = []
    for i in range(N):
        # if i % 100==0:
        #     print(f'Processing {i}/{N}')
        pathN = len(data[i])
        normed_path = []
        for j in range(pathN):
            normed_path.append([
                # (data[i][j][0] - x_min) / scale,
                # (data[i][j][1] - y_min) / scale
                (data[i][j][0] + (scale - x_max - x_min) / 2) / scale,
                (data[i][j][1] + (scale - y_max - y_min) / 2) / scale
            ])
        normed_data.append(normed_path)

    return normed_data, scale, x_min, y_min, x_max, y_max

def normalize_data3(pos, scale, x_min, y_min, x_max, y_max):
    x, y = pos
    normed_pos= [
        (x + (scale - x_max - x_min) / 2) / scale,
        (y + (scale - y_max - y_min) / 2) / scale
    ]
    return normed_pos


def remove_short_edges(data, short_threshold=0.3, indices=None):
    N = len(data)

    filtered_data = []
    filtered_idx = []

    for i in tqdm(range(N), desc="Removing short edges"):
        N_path = len(data[i])
        start = np.array(data[i][0])
        end = np.array(data[i][N_path - 1])
        if np.linalg.norm(start - end) > short_threshold:
            filtered_data.append(data[i])
            if indices is not None:
                filtered_idx.append(indices[i])

    if indices is not None:
        return filtered_data, filtered_idx
    else:
        return filtered_data

def remove_zig_zag_path(data):
    N = len(data)

    filtered_data = []

    for i in tqdm(range(N), desc="Removing zig-zag path"):
        N_path = len(data[i])
        flag = True

        for j in range(1, N_path - 1):
            pos_pre = np.array(data[i][j - 1])
            pos_cur = np.array(data[i][j])
            pos_next = np.array(data[i][j + 1])

            dir_0 = pos_cur - pos_pre
            dir_1 = pos_next - pos_cur

            if np.linalg.norm(dir_0) < 1e-5 or np.linalg.norm(dir_1) < 1e-5:
                continue

            dir_0 = dir_0 / np.linalg.norm(dir_0)
            dir_1 = dir_1 / np.linalg.norm(dir_1)

            # if dir_0.dot(dir_1) < 0.5:
            if dir_0.dot(dir_1) < 0:
                flag = False
                break
        
        if flag:
            filtered_data.append(data[i])
    
    return filtered_data

def simplify_zig_zag_path(data, indices=None):
    N = len(data)
    simplified_path = []
    simplified_indices = []

    for i in tqdm(range(N), desc="Simplifying zig-zag path"):
        N_path = len(data[i])
        path = []
        for j in range(N_path):
            if j == 0 or j == N_path - 1:
                path.append(data[i][j])
            else:
                prev = np.array(path[-1])
                cur = np.array(data[i][j])
                next = np.array(data[i][j + 1])
                if np.linalg.norm(prev - cur) < 1e-5 or np.linalg.norm(next - cur) < 1e-5:
                    continue
                
                dir_0 = cur - prev
                dir_0 = dir_0 / np.linalg.norm(dir_0)
                dir_1 = next - cur
                dir_1 = dir_1 / np.linalg.norm(dir_1)

                if dir_0.dot(dir_1) < -0.9:
                    continue
                    
                path.append(data[i][j])
        if len(path) >= 2:
            simplified_path.append(path)
            if indices is not None:
                simplified_indices.append(indices[i])
    
    if indices is not None:
        return simplified_path, simplified_indices
    else:
        return simplified_path

def remove_repeat_point(data):
    N = len(data)

    filtered_data = []

    for i in tqdm(range(N), desc="Removing repeat point"):
        N_path = len(data[i])
        path = []
        for j in range(N_path):
            if j == 0 or np.linalg.norm(np.array(data[i][j]) - np.array(data[i][j - 1])) > 1e-5:
                path.append(data[i][j])
        filtered_data.append(path)
    
    return filtered_data

def sample_single_path(path, num_samples=5):
    line = LineString(path)
    distances = [line.length * i / (num_samples - 1) for i in range(num_samples)]
    sampled_points = [line.interpolate(distance) for distance in distances]
    return [[point.x, point.y] for point in sampled_points]

def sample_path(paths, num_samples=5):
    N = len(paths)
    sampled_paths = []

    for i in tqdm(range(N), desc="Sampling paths"):
        sampled_paths.append(sample_single_path(paths[i], num_samples))
    
    return sampled_paths

def create_colormap(colors, n_segments):
    color_start = colors[0]
    color_end = colors[1]

    cur_colors = []
    for i in range(n_segments):
        t = i / (n_segments - 1)
        cur_color_interp = tuple(np.interp(t, [0, 1], [color_start[j], color_end[j]]) for j in range(4))
        cur_colors.append(cur_color_interp)

    return cur_colors

def split_segments(segments, split_dis=0.05):
    N = len(segments)
    new_segments = []

    for segment in segments:
        start = np.array(segment[0])
        end = np.array(segment[1])
        if np.linalg.norm(start - end) < split_dis:
            new_segments.append([(start[0], start[1]), (end[0], end[1])])
        else:
            split_num = int(np.ceil(np.linalg.norm(start - end) / split_dis))
            for i in range(split_num):
                t_0 = i / split_num
                t_1 = (i + 1) / split_num
                cur_start = start + t_0 * (end - start)
                cur_end = start + t_1 * (end - start)
                new_segments.append([(cur_start[0], cur_start[1]), (cur_end[0], cur_end[1])])
    
    return new_segments

def save_data_input_image(filename, path_list):
    file_path = f'./log/image/Input/{filename}.png'
    plt.rcParams['figure.figsize'] = (4, 4)
    
    # 定义渐变的颜色列表（RGBA格式）
    start_color = (1, 0, 0, 0.2)  # 红色
    end_color = (0, 0, 1, 0.2)  # 蓝色
    
    # 创建颜色映射
    colors = [start_color, end_color]
    
    # normalize data
    # path_list = normalize_data(path_list)
    # 创建一个图形对象和轴对象
    fig, ax = plt.subplots()
    
    lines = []
    
    # 遍历路径列表并绘制渐变折线
    for i, path_data in enumerate(path_list):
        # 提取x和y坐标
        path_data = np.array(path_data)
        x = path_data[:, 0]
        y = path_data[:, 1]
        
        # points = np.column_stack((x, y))
        segments = [[(x[i], y[i]), (x[i + 1], y[i + 1])] for i in range(len(path_data) - 1)]
        segments = split_segments(segments)

        cur_colors = create_colormap(colors, len(segments) + 1)
        
        # 创建LineCollection对象，并使用渐变颜色
        lc = LineCollection(segments, colors=cur_colors, linewidth=0.2)
        lc.set_array(np.arange(len(path_list)))
        ax.add_collection(lc)
        lines.append(lc)

        radius = 6.0 / 1024
        
        # 添加起点和终点的标记
        ax.add_patch(Circle((x[0], y[0]), radius=radius, color=(0.2, 0.2, 0.2), lw=0.2))
        # 添加红色标号
        # ax.text(x[0], y[0], f'{x[0]:.2f},{y[0]:.2f}', fontsize=2, color='red')
        ax.text(x[0], y[0], f'{i}', fontsize=2, color='red')
        ax.add_patch(Circle((x[-1], y[-1]), radius=radius, fill=False, color=(0.2, 0.2, 0.2), lw=0.2))
        # 添加蓝色标号
        # ax.text(x[-1], y[-1], f'{x[-1]:.2f},{y[-1]:.2f}', fontsize=2, color='blue')
        ax.text(x[-1], y[-1], f'{i}', fontsize=2, color='blue')
    
    ax.text(0, 0, f'{filename}({len(path_list)})', color='blue')

    # 设置轴的范围
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    
    # 设置x和y坐标的缩放一样
    ax.set_aspect('equal')
    
    # # 添加颜色条
    # cbar = plt.colorbar(lines[0], ax=ax, orientation='vertical', ticks=np.arange(len(path_list)))
    # cbar.set_label('Path')
    
    # 保存图像为pic.png，并指定分辨率
    plt.savefig(file_path, dpi=1000)
    
    # 关闭图形窗口
    plt.close()


def save_trend_image(filename, path_list):
    file_path = f'./log/image/Input/gtTrend/{filename}.png'
    plt.rcParams['figure.figsize'] = (4, 4)

    # 定义渐变的颜色列表（RGBA格式）
    start_color = (1, 0, 0, 0.2)  # 红色
    end_color = (0, 0, 1, 0.2)  # 蓝色

    # 创建颜色映射
    colors = [start_color, end_color]

    # normalize data
    # path_list = normalize_data(path_list)
    # 创建一个图形对象和轴对象
    fig, ax = plt.subplots()

    lines = []

    # 遍历路径列表并绘制渐变折线
    for i, path_data in enumerate(path_list):
        # 提取x和y坐标
        path_data = np.array(path_data)
        x = path_data[:, 0]
        y = path_data[:, 1]

        # points = np.column_stack((x, y))
        segments = [[(x[i], y[i]), (x[i + 1], y[i + 1])] for i in range(len(path_data) - 1)]
        segments = split_segments(segments)

        cur_colors = create_colormap(colors, len(segments) + 1)

        # 创建LineCollection对象，并使用渐变颜色
        lc = LineCollection(segments, colors=cur_colors, linewidth=3)
        lc.set_array(np.arange(len(path_list)))
        ax.add_collection(lc)
        lines.append(lc)

        radius = 6.0 / 1024

        # 添加起点和终点的标记
        ax.add_patch(Circle((x[0], y[0]), radius=radius, color=(0.2, 0.2, 0.2), lw=0.2))
        # 添加红色标号
        # ax.text(x[0], y[0], f'{x[0]:.2f},{y[0]:.2f}', fontsize=2, color='red')
        ax.text(x[0], y[0], f'{i}', fontsize=2, color='red')
        ax.add_patch(Circle((x[-1], y[-1]), radius=radius, fill=False, color=(0.2, 0.2, 0.2), lw=0.2))
        # 添加蓝色标号
        # ax.text(x[-1], y[-1], f'{x[-1]:.2f},{y[-1]:.2f}', fontsize=2, color='blue')
        ax.text(x[-1], y[-1], f'{i}', fontsize=2, color='blue')

    # 设置轴的范围
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    # 设置x和y坐标的缩放一样
    ax.set_aspect('equal')

    # # 添加颜色条
    # cbar = plt.colorbar(lines[0], ax=ax, orientation='vertical', ticks=np.arange(len(path_list)))
    # cbar.set_label('Path')

    # 保存图像为pic.png，并指定分辨率
    plt.savefig(file_path, dpi=1000)

    # 关闭图形窗口
    plt.close()

def draw_ground_truth(filename):
    # remove "_packed" from the filename
    filename = filename.replace('_packed', '')
    with open(PATH + 'generated/groundtruth/' + filename + '.pkl', 'rb') as file:
        data = pickle.load(file)
    curve = data['curve']
    print(curve)
    save_trend_image(filename, [curve])

    control_points = data['control_points']
    start_center = data['start_center']
    end_center = data['end_center']
    direction = (np.array(end_center) - np.array(start_center)) / np.linalg.norm(np.array(end_center) - np.array(start_center))
    orthogonal = np.array([-direction[1], direction[0]])
    if len(control_points) == 1:
        origin_control_points = control_points.copy()
        control_points[0] = np.array(origin_control_points[0]) + 0.25 * orthogonal
        curve1 = get_bspline_curve(start_point=start_center, end_point=end_center, control_points=control_points, samples=100, k=2)
        save_trend_image(filename + '_alter1', [curve1])
        control_points[0] = np.array(origin_control_points[0]) - 0.25 * orthogonal
        curve2 = get_bspline_curve(start_point=start_center, end_point=end_center, control_points=control_points, samples=100, k=2)
        save_trend_image(filename + '_alter2', [curve2])
        if np.random.rand() < 0.5:
            control_points[0] = np.array(origin_control_points[0] + 0.5 * orthogonal)
        else:
            control_points[0] = np.array(origin_control_points[0] - 0.5 * orthogonal)
        curve3 = get_bspline_curve(start_point=start_center, end_point=end_center, control_points=control_points, samples=100, k=2)
        save_trend_image(filename + '_alter3', [curve3])
    if len(control_points) == 2:
        origin_control_points = control_points.copy()
        if np.random.rand() < 0.5:
            control_points[0] = np.array(origin_control_points[0]) + 0.25 * orthogonal
        else:
            control_points[0] = np.array(origin_control_points[0]) - 0.25 * orthogonal
        if np.random.rand() < 0.5:
            control_points[1] = np.array(origin_control_points[1]) + 0.25 * orthogonal
        else:
            control_points[1] = np.array(origin_control_points[1]) - 0.25 * orthogonal
        curve1 = get_bspline_curve(start_point=start_center, end_point=end_center, control_points=[origin_control_points[0], control_points[1]], samples=100, k=2)
        save_trend_image(filename + '_alter1', [curve1])
        curve2 = get_bspline_curve(start_point=start_center, end_point=end_center, control_points=[control_points[0], origin_control_points[1]], samples=100, k=2)
        save_trend_image(filename + '_alter2', [curve2])
        curve3 = get_bspline_curve(start_point=start_center, end_point=end_center, control_points=control_points, samples=100, k=2)
        save_trend_image(filename + '_alter3', [curve3])


if __name__ == '__main__':
    # dataset_name = 'DenmarkCoast_new_concat'
    # data = load_data(dataset_name)
    # save_data_input_image(dataset_name, data)
    dataset_name = 'trend_0.0_0.90_-0.81_0.59_1.00_1_1_1'
    datasets = [
        'trend_0.0_0.90_-1.00_0.00_1.17_1_1_1_packed',  # [13,15,10]
        'trend_0.0_0.90_-0.81_0.59_1.04_1_1_2_packed',  # [15,16,21]
        'trend_0.0_0.90_-0.81_0.59_1.12_1_2_1_packed',  # [11,12,17]
        'trend_0.0_0.90_-0.81_0.59_1.00_2_1_1_packed',  # [14,15,18]
        'trend_0.0_0.90_-0.81_0.59_1.08_2_1_2_packed',  # [16,18,20]
        'trend_0.0_0.90_-0.81_0.59_1.21_2_2_1_packed',  # [17,10,20]
    ]
    for dataset_name in datasets:
        draw_ground_truth(dataset_name)

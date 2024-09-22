import numpy as np
from scipy.interpolate import BSpline, CubicSpline
import matplotlib.pyplot as plt
import time

def get_knot_by_dis(path):
    N_path = len(path)
    path = np.array(path)
    total_length = 0
    for i in range(N_path - 1):
        total_length += np.linalg.norm(path[i] - path[i + 1])
    
    cur_dis = 0
    t = []
    for i in range(N_path):
        t.append(cur_dis / total_length)
        if i < N_path - 1:
            cur_dis += np.linalg.norm(path[i] - path[i + 1])
    t = np.array(t)

    return t

def interpolate_knot(t, inter_num):
    new_t = []

    N_path = len(t)
    for i in range(N_path):
        if i > 0:
            pre_t = t[i - 1]
            cur_t = t[i]
            t_step = (cur_t - pre_t) / inter_num
            for j in range(1, inter_num + 1):
                new_t.append(pre_t + t_step * j)
        else:
            new_t.append(t[i])
    
    new_t = np.array(new_t)
    return new_t

def get_clamped_cubic_spline(path, inter_num, start_tangent, end_tangent, clamped_scale=0.1):
    N_path = len(path)
    path = np.array(path)
    # 定义数据点
    t = get_knot_by_dis(path)  # 参数值
    x = path[:, 0]  # x坐标
    y = path[:, 1]  # y坐标

    # 给定起点和终点的方向向量
    start_tangent = np.array(start_tangent) * clamped_scale  # 起点处的方向向量
    end_tangent = np.array(end_tangent)  * clamped_scale # 终点处的方向向量

    if np.linalg.norm(start_tangent) < 1e-5:
        start_condition_x = 'natural'
        start_condition_y = 'natural'
    else:
        start_condition_x = (1, start_tangent[0])
        start_condition_y = (1, start_tangent[1])
    
    if np.linalg.norm(end_tangent) < 1e-5:
        end_condition_x = 'natural'
        end_condition_y = 'natural'
    else:
        end_condition_x = (1, end_tangent[0])
        end_condition_y = (1, end_tangent[1])
    
    # avoid same values in t
    # add a 0.0001x on t
    for i in range(len(t)):
        t[i] += 0.000001 * i / len(t)

    try:
        # 对 x(t) 和 y(t) 分别进行克拉克样条插值
        cs_x = CubicSpline(t, x, bc_type=(start_condition_x, end_condition_x))
        cs_y = CubicSpline(t, y, bc_type=(start_condition_y, end_condition_y))
    except Exception as e:
        print(f'cubic spline interpolation error: {e}')
        print(f't : {t}')
        print(f'x : {x}')
        raise Exception(f'cubic spline interpolation error: {e}')

    # 使用插值对象计算新的点
    t_new = interpolate_knot(t, inter_num)
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)

    
    # # 绘制结果
    # plt.figure(figsize=(10, 10))
    # plt.plot(x, y, 'o')
    # plt.plot(x_new, y_new)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis('equal')  # 确保x和y轴的单位长度相同
    # # plt.show()
    # plt.savefig(f'./log/image/Smooth/{time.perf_counter()}.png')
    # plt.close()

    return np.vstack((x_new, y_new)).T

def get_bspline_curve(start_point, control_points, end_point, samples=100, k=3):
    # 计算参数化节点向量
    # B样条的阶数k（次数为k-1）
    c = len(control_points) + 2 # 控制点数量
    # print(f'k={k}, c={c}')
    t = np.concatenate(([0 for _ in range(k)], np.linspace(0, 1, c - k + 1), [1 for _ in range(k)]))
    # print(f't={t}')

    stacked_points = np.vstack((start_point, control_points, end_point))
    # 创建B样条曲线对象
    bspline = BSpline(t, stacked_points, k, extrapolate=False)

    # 生成采样点
    u = np.linspace(0, 1, samples)
    curve = bspline(u)
    
    return curve

def get_straight_line(start_point, end_point, samples):
    # compute the intermediate position
    line = np.zeros((samples, 2))
    for i in range(samples):
        frac = i / (samples - 1)
        line[i] = start_point * (1 - frac)
        line[i] += end_point * frac
    return line

def calculate_distance(point1, point2):
    """Calculate the distance between two points."""
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def total_path_length(path):
    """Calculate the total length of the path."""
    length = 0
    for i in range(len(path) - 1):
        length += calculate_distance(path[i], path[i+1])
    return length

def interpolate_path(path, ratio):
    if ratio <= 0:
        return path[0]
    elif ratio >= 1:
        return path[-1]
    
    length = total_path_length(path)
    cur_length = length * ratio

    for i in range(len(path) - 1):
        segment_length = np.linalg.norm(path[i] - path[i + 1])
        if cur_length <= segment_length:
            t = cur_length / segment_length
            new_point = path[i]+ t * (path[i + 1] - path[i])
            return np.array(new_point)
        else:
            cur_length -= segment_length
    
    return path[-1]

def get_origin_path_sample(path, num_samples):
    """Sample the path at equal intervals, ensuring the inclusion of start and end points."""
    copied_path = path.copy()

    total_length = total_path_length(copied_path)
    interval = total_length / (num_samples - 1)
    sampled_path = [path[0]]  # Start with the first point
    current_length = 0
    accumulated_length = 0  # Total length accumulated over segments
    
    for i in range(len(copied_path) - 1):
        segment_length = calculate_distance(copied_path[i], copied_path[i+1])
        while accumulated_length + segment_length >= interval:
            overshoot = interval - accumulated_length
            ratio = overshoot / segment_length
            new_point = [copied_path[i][0] + ratio * (copied_path[i+1][0] - copied_path[i][0]),
                         copied_path[i][1] + ratio * (copied_path[i+1][1] - copied_path[i][1])]
            sampled_path.append(new_point)
            # Adjust the starting point of the current segment to the new point
            copied_path[i] = new_point
            # Recalculate the remaining length of the current segment
            segment_length = calculate_distance(copied_path[i], copied_path[i+1])
            accumulated_length = 0  # Reset accumulated length for the next interval
        
        accumulated_length += segment_length  # Accumulate the length of the current segment

    # If the last point wasn't added, add it
    if len(sampled_path) < num_samples:
        sampled_path.append(path[-1])

    return sampled_path


def resample_path(path, frames):
    """Resample the path to the specified number of frames."""
    num_samples = len(path)
    if num_samples == frames:
        return path
    sampled_path = get_origin_path_sample(path, frames)
    return sampled_path


if __name__ == '__main__':
    path = [[0, 3], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]]
    start_tangent = [0, 0]
    end_tangent = [0, 0]
    get_clamped_cubic_spline(path, 100, start_tangent, end_tangent, 5)

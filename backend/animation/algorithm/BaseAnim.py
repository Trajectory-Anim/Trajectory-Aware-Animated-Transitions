import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
# from utils.showanim.showanim import show_animation
from utils.showanimcpp.showanimcpp import show_animation
from utils.interpolate.interpolate import get_bspline_curve, get_straight_line, get_origin_path_sample, interpolate_path, resample_path
from animation.AnimData import *
from pytweening import *

# Base Anim is an abstract class(ABC) for grid animation.
class BaseAnim(ABC):
    class PathMode(Enum):
        BUNDLE = 0
        STRAIGHT = 1
        CUSTOM = 2
        ORIGIN_PATH = 3
    
    class MovementMode(Enum):
        TOGETHER = 0
        ORDER = 1
        CUSTOM = 2
    
    class RadiusMode(Enum):
        SAME = 0
        WEIGHT = 1

    def __init__(self, T, r, path_mode, movement_mode, radius_mode, algo_name):
        # input dataset
        self.start_position = None          # the start position of each grid
        self.end_position = None            # the end position of each grid
        self.path_position = None          # the path position of each grid
        self.N = None                       # the amount of the grid
        # animation parameters
        assert T > 0
        self.T = T  # the time for each group
        self.r = r  # the radius of each grid
        self.path_mode = path_mode  # bundle or straight or custom
        self.movement_mode = movement_mode  # together or order or custom
        self.radius_mode = radius_mode  # same or weight
        self.algo_name = algo_name  # the name of the algorithm

        # generate
        self.amount_groups = None  # the amount of groups
        self.anime_groups = None  # the groups of each grid. groups[i, j] == 1 if grid i is in group j
        self.anime_paths = None  # the path of each grid

        # results
        self.anime_position = None          # the position of each grid at each time point (include start and end)
        self.anime_start = None             # the start time point of each grid
        self.anime_end = None               # the end time point of each grid

    def set_input(self, dataset_name, paths):
        # set input
        self.dataset_name = dataset_name
        self.path_position = paths
        self.start_position = np.array([path[0] for path in paths])
        self.end_position = np.array([path[-1] for path in paths])
        self.N = len(paths)

        # load data weight
        try:
            self.path_weight = load_data_weight(dataset_name)
            self.path_weight = np.array(self.path_weight, dtype=np.float32)
        except:
            self.path_weight = np.ones(self.N, dtype=np.float32)
        # normalize
        self.path_weight = self.path_weight / np.max(self.path_weight)

        # only for test
        self.path_weight = np.ones(self.N, dtype=np.float32)

        if self.radius_mode == self.RadiusMode.WEIGHT:
            # self.radii = np.sqrt(self.path_weight)
            weights = np.log(self.path_weight + 1)
            normalized_weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
            # if min(weights) == max(weights), then all weights are the same, set radii to 1
            if np.min(weights) == np.max(weights):
                self.radii = np.ones(self.N, dtype=np.float32)
            else:
                # set 4 different radii levels: 1, 2, 3, 4, according to the weights
                # top 25% are 4, 25%-50% are 3, 50%-75% are 2, 75%-100% are 1
                self.radii = np.zeros(self.N, dtype=np.float32)
                for i in range(self.N):
                    if normalized_weights[i] < 0.25:
                        self.radii[i] = 1
                    elif normalized_weights[i] < 0.5:
                        self.radii[i] = 1.5
                    elif normalized_weights[i] < 0.75:
                        self.radii[i] = 2
                    else:
                        self.radii[i] = 2.5


            # self.radii = self.radii / np.max(self.radii) * 2
            # self.radii = np.zeros(self.N, dtype=np.float32)


            # for i in range(self.N):
            #     if self.radii[i] < 1:
            #         self.radii[i] = 1
            # self.radii = self.radii / np.min(self.radii)

        else:
            self.radii = np.ones(self.N, dtype=np.float32)

        self.all_name = f'{self.dataset_name}_{self.algo_name}'
    
    def init_animation(self):
        # init groups and paths
        self.anime_groups = np.zeros((self.N, self.amount_groups), dtype=np.int32)
        self.anime_start = np.zeros((self.N), dtype=np.int32)
        self.anime_end = np.ones((self.N), dtype=np.int32) * (self.T + 1)

    @abstractmethod
    def compute_group(self):
        # implement the function to compute the group of each grid
        pass

    def ease_start_and_end(self, t):
        ease_ratio = self.get_ease_ratio()
        ease_part = 0.2
        eased_sum_t = (1 - 2 * ease_part) + 2 * ease_part * ease_ratio
        eased_t = t * eased_sum_t
        
        if eased_t < ease_part * ease_ratio:
            return easeInSine(eased_t / (ease_part * ease_ratio)) * ease_part
        elif eased_t < ease_part * ease_ratio + (1 - 2 * ease_part):
            return eased_t - ease_part * ease_ratio + ease_part
        else:
            tmp = (eased_t - ease_part * ease_ratio - (1 - 2 * ease_part)) / (ease_part * ease_ratio)
            if tmp > 1:
                tmp = 1
            return easeOutSine(tmp) * ease_part + ease_part + (1 - 2 * ease_part)
    
    def apply_slow_in_out(self, path):
        new_path = []
        N_path = len(path)
        for i in range(N_path):
            t = i / (N_path - 1)
            # eased_t = self.ease_func(t)
            eased_t = self.ease_start_and_end(t)
            new_path.append(interpolate_path(path, eased_t))
        
        return np.array(new_path)

    def get_ease_ratio(self):
        step = 1e-5
        mid = 0.5
        return (self.ease_func(mid + step) - self.ease_func(mid)) / step;

    def compute_path(self):
        self.anime_paths = np.zeros((self.N, self.T, 2))

        # if using custom path, the method must provide the anime_paths
        if self.path_mode == self.PathMode.CUSTOM:
            raise Exception('Please override the compute_path method')
        
        elif self.path_mode == self.PathMode.STRAIGHT:
            for point in range(self.N):
                self.anime_paths[point] = get_straight_line(self.start_position[point], self.end_position[point], self.T)

        elif self.path_mode == self.PathMode.ORIGIN_PATH:
            for point in range(self.N):
                self.anime_paths[point] = get_origin_path_sample(self.path_position[point], self.T)

        elif self.path_mode == self.PathMode.BUNDLE:
            for clusteridx in range(0, self.amount_groups):
                group = [point for point in range(self.N) if self.anime_groups[point, clusteridx] == 1]
                # get three control points of B-Spline Curve
                start_control = np.average(self.start_position[group], axis=0)
                end_control = np.average(self.end_position[group], axis=0)
                center_control = (start_control + end_control) / 2
                control_points = np.array([start_control, center_control, end_control])

                for point in range(self.N):
                    if self.anime_groups[point, clusteridx] == 1:
                        # generate the curve according to path mode
                        self.anime_paths[point] = get_bspline_curve(self.start_position[point], control_points, self.end_position[point], self.T)

    def compute_animation(self):
        # the general function to combine the animation of each group

        if self.movement_mode == self.MovementMode.TOGETHER:
            avg_travel_distance = 0
            for point in range(self.N):
                for t in range(self.T-1):
                    avg_travel_distance += np.linalg.norm(self.anime_paths[point][t] - self.anime_paths[point][t+1])
            avg_travel_distance /= self.N
            new_T = int(avg_travel_distance/0.3 * 60)
            print(new_T)

            # move together
            self.anime_position = np.zeros((new_T + 2, self.N, 2))

            self.anime_position[0] = self.start_position
            self.anime_position[new_T + 1] = self.end_position
            for clusteridx in range(0, self.amount_groups):
                group = [point for point in range(self.N) if self.anime_groups[point, clusteridx] == 1]
                for point in range(self.N):
                    if point in group:
                        curve = self.anime_paths[point]
                        new_curve = resample_path(curve, new_T)
                        new_curve = np.array(new_curve)

                        self.anime_start[point] = 0
                        self.anime_end[point] = new_T + 1
                        for t in range(new_T):
                            self.anime_position[t+1][point] = new_curve[t]

        elif self.movement_mode == self.MovementMode.ORDER:
            self.ease_func = easeInOutSine
            ease_ratio = self.get_ease_ratio()

            avg_travel_distance_per_group = []
            for clusteridx in range(0, self.amount_groups):
                group = [point for point in range(self.N) if self.anime_groups[point, clusteridx] == 1]
                total_travel_distance = 0
                for point in group:
                    for t in range(self.T-1):
                        total_travel_distance += np.linalg.norm(self.anime_paths[point][t] - self.anime_paths[point][t+1])
                avg_travel_distance_per_group.append(total_travel_distance / len(group))
            T_for_each_group = []
            for clusteridx in range(0, self.amount_groups):
                T_for_each_group.append(int(avg_travel_distance_per_group[clusteridx]/0.3 * 60 * ease_ratio))
            print(T_for_each_group)
            # move by order
            self.anime_position = np.zeros((sum(T_for_each_group) + 2, self.N, 2))
            # set the start and end position
            self.anime_position[0] = self.start_position
            self.anime_position[sum(T_for_each_group) + 1] = self.end_position
            for clusteridx in range(0, self.amount_groups):
                group = [point for point in range(self.N) if self.anime_groups[point, clusteridx] == 1]
                for point in range(self.N):
                    if point in group:
                        curve = self.anime_paths[point]
                        new_curve = resample_path(curve, T_for_each_group[clusteridx])
                        new_curve = np.array(new_curve)
                        new_curve = self.apply_slow_in_out(new_curve)
                        self.anime_start[point] = sum(T_for_each_group[:clusteridx]) + 1
                        self.anime_end[point] = self.anime_start[point] + T_for_each_group[clusteridx] - 1
                        for t in range(T_for_each_group[clusteridx]):
                            self.anime_position[sum(T_for_each_group[:clusteridx]) + t + 1][point] = new_curve[t]
                    else:
                        for t in range(T_for_each_group[clusteridx]):
                            self.anime_position[sum(T_for_each_group[:clusteridx]) + t + 1][point] = self.anime_position[sum(T_for_each_group[:clusteridx]) + t][point]
            
            # if self.path_mode == self.PathMode.ORIGIN_PATH:
                # # vertical flip anime_position and rotate 90 degree with center (0.5, 0.5)
                # for t in range(sum(T_for_each_group) + 2):
                #     for point in range(self.N):
                #         self.anime_position[t][point] = np.array([1-self.anime_position[t][point][1], 1 - self.anime_position[t][point][0]])
            # else:
            # vertical flip anime_position
            # for t in range(sum(T_for_each_group) + 2):
            #     for point in range(self.N):
            #         self.anime_position[t][point] = np.array([1-self.anime_position[t][point][0], 1 - self.anime_position[t][point][1]])



            # # move by order
            # self.anime_position = np.zeros((self.amount_groups * self.T + 2, self.N, 2))
            # # set the start and end position
            # self.anime_position[0] = self.start_position
            #
            # self.anime_position[self.amount_groups * self.T + 1] = self.end_position
            # for clusteridx in range(0, self.amount_groups):
            #     group = [point for point in range(self.N) if self.anime_groups[point, clusteridx] == 1]
            #
            #     for point in range(self.N):
            #         if point in group:
            #             curve = self.anime_paths[point]
            #             curve = self.apply_slow_in_out(curve)
            #
            #             self.anime_start[point] = clusteridx * self.T + 1
            #             self.anime_end[point] = self.anime_start[point] + self.T - 1
            #
            #             for t in range(self.T):
            #                 anime_index = clusteridx * self.T + t + 1
            #                 self.anime_position[anime_index][point] = curve[t]
            #
            #         else:
            #             for t in range(self.T):
            #                 anime_index = clusteridx * self.T + t + 1
            #                 self.anime_position[anime_index][point] = self.anime_position[anime_index - 1][point]
        else:
            raise Exception('Please override the compute_animation method')

    def show(self, dataset_name, trend_mode=True, points_to_trace=None):
        # print(f'self.r={self.r}')
        # print(f'self.radii={self.radii}')
        if points_to_trace == None:
            points_to_trace = [1]

        if trend_mode:
            grid_n = 8
            points_to_trace = []
            mode_str = 'TREND'
        else:
            grid_n = 0
            mode_str = 'TRACE'
        

        show_animation(
            self.anime_position,
            self.anime_groups,
            self.anime_start,
            self.anime_end,
            self.r*self.radii,
            1024,
            1024,
            self.algo_name,
            60,
            0.05,
            1,
            # f'Animation/NORMAL_{dataset_name}_{self.algo_name}.mp4',
            # f'Animation/{mode_str}_{dataset_name}_{self.algo_name}.mp4',
            f'Animation/{dataset_name}_{self.algo_name}.mp4',
            2,
            [],
            True,
            points_to_trace,
            grid_n
            # False
        )
    
    def save_to_svg(self, dataset_name, interval=10, target_group=None):
        import svgwrite
        import os

        # 创建保存SVG文件的目录
        save_dir = f'log/SVG/{dataset_name}_{self.algo_name}'
        os.makedirs(save_dir, exist_ok=True)

        # 获取动画总帧数
        total_frames = self.anime_position.shape[0]

        for frame in range(0, total_frames, interval):
            # 创建SVG文件
            dwg = svgwrite.Drawing(f'{save_dir}/frame_{frame:04d}.svg', size=('1024px', '1024px'))
            # 添加浅灰色背景
            dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='#f8f8f8'))

            def check_highlight(point):
                if target_group is not None:
                    return self.anime_groups[point, target_group] == 1
                else:
                    return self.anime_start[point] <= frame <= self.anime_end[point]

            # 绘制黑色点的路径
            if hasattr(self, 'bundled_position'):
                path_color = "#cccccc"
                path_opacity = 1
                path_array = self.bundled_position
            else:
                path_color = 'black'
                path_opacity = 0.1
                path_array = self.anime_position

            for point in range(self.N):
                if check_highlight(point):
                    path = path_array[self.anime_start[point]:self.anime_end[point]+1, point]
                    path_points = [(x*1024, y*1024) for x, y in path]
                    
                    # 创建SVG路径
                    svg_path = dwg.path(d=f"M {path_points[0][0]} {path_points[0][1]}")
                    for x, y in path_points[1:]:
                        svg_path.push(f"L {x} {y}")
                    
                    # 设置路径样式
                    svg_path.stroke(color=path_color, width=2, opacity=path_opacity)
                    svg_path.fill(color="none")
                    
                    # 添加路径到SVG
                    dwg.add(svg_path)

            # 只绘制高亮(黑色)点
            for point in range(self.N):
                if check_highlight(point):
                    x, y = self.anime_position[frame][point]
                    radius = self.r * self.radii[point] * 1024
                    dwg.add(dwg.circle(center=(x*1024, y*1024), r=radius, fill='black'))

            # 保存SVG文件
            dwg.save()

        print(f"SVG文件已保存到 {save_dir} 目录")
        
        # show_animation(
        #     self.anime_position,
        #     self.anime_groups,
        #     self.anime_start,
        #     self.anime_end,
        #     radius=self.r*self.radii,
        #     title=self.algo_name,
        #     save_to_file_name=f'Animation/{dataset_name}_{self.algo_name}.mp4')

    def get_output(self):
        self.compute_group()
        self.compute_path()
        self.compute_animation()
        return self.anime_position

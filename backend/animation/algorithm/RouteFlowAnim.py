from animation.algorithm.BaseAnim import BaseAnim
from animation.AnimData import *
import numpy as np
import cv2
from utils.showanim.showanim import get_color, save_animation_to_file
from utils.Bundlesolver import Bundlesolver
from utils.Animsolver import Animsolver
from utils.Groupsolver import Groupsolver
from utils.interpolate.interpolate import *
import time
import json
import os
import copy

class RouteFlowAnim(BaseAnim):

    def __init__(self, T=110, r=6.0/1024,
                 # Bundling parameters
                 bundle_algo='HEB', n_stage=2, n_iter=50, dt=1e-4, d=0.1, md_list=[0.02, 0.1], move_lim=0.5,
                 connectivity_strength=3, connectivity_angle=120, merge_iter=10,
                 kNN_k=5,
                 k_s=5, k_c=1, k_a=1, s_list=[0.005, 0.02],
                 k_c_rate_list=[1, 1],
                 subgroup_dis=0.075,
                 # Grouping parameters
                 n_c=1e-4,
                 # Packing and animation parameters
                 eps=1e-5, key_point_merge_dis=0.05, start_padding_dis=-1,
                 base_velocity=0.4, trans_speed=1.0, sedimentation_speed=1e5,
                 slowdown_topk=0.1,
                 # DAG merging parameters
                 DAG_merge_eps=0.1, DAG_merge_checkdot=0.5,
                 # Interpolation parameters
                 inter_num=100, ease_function="Cubic", ease_ratio=0.1,
                 debug_mode=False, only_bundle_mode=False, quit_after_packing=False):
        # set up all the parameters
        params = locals()
        params.pop('self')
        for key, value in params.items():
            setattr(self, key, value)
        
        # get algorithm name
        self.name_bundle = f'(0912)ks{k_s}_ka{k_a}_kc{k_c}'

        self.name_grouping = f'nc{n_c}'

        name = f'RF_{self.name_bundle}_{self.name_grouping}'

        super().__init__(T, r, BaseAnim.PathMode.CUSTOM, BaseAnim.MovementMode.CUSTOM, BaseAnim.RadiusMode.SAME, name)

    def set_input(self, dataset_name, paths):
        self.all_name_bundle = f'{dataset_name}_{self.name_bundle}'
        super().set_input(dataset_name, paths)

    def compute_control_point(self):
        # resample paths
        for i in range(self.N):
            self.path_position[i] = resample_path(self.path_position[i], 10)


        print('Computing Control Points...')

        final_bundled_path_start = [[self.path_position[i][0]] for i in range(self.N)]
        final_bundled_path_mid = [[] for i in range(self.N)]
        final_bundled_path_end = [[self.path_position[i][-1]] for i in range(self.N)]
        final_groups = []

        self.all_stages_path_history = []
        self.all_stages_group_history = []
        last_finished_group = []
        cur_finished_group = []
        self.all_stages_result_path_history = []

        cur_sub_groups = [[i] for i in range(self.N)]
        cur_sub_groups_path = self.path_position

        bundle_time_total = 0

        for cur_stage in range(1, self.n_stage + 1):
            # print(f'[DEBUG] cur_stage = {cur_stage}')
            # print(f'[DEBUG] len of cur_sub_groups_path = {len(cur_sub_groups_path)}')
            # print each path on each line
            # for i, path in enumerate(cur_sub_groups_path):
            #     print(f'[DEBUG] path {i} = {path}')

            time_point1 = time.time()
            bundled_sub_groups_path_history = Bundlesolver.get_control_points_lists(self, cur_sub_groups_path, cur_stage)
            time_point2 = time.time()
            bundle_time_total += time_point2 - time_point1

            # remove bundled path's start and end
            bundled_sub_groups_path = copy.deepcopy(bundled_sub_groups_path_history[-1])
            # print(f'[DEBUG] bundled_sub_groups_path = {bundled_sub_groups_path}')

            sub_group_one_hot = Groupsolver.get_group(self, bundled_sub_groups_path, cur_stage)
            # print(f'[DEBUG] sub_group_one_hot = {sub_group_one_hot}')

            N_sub_group = len(sub_group_one_hot)
            N_group = len(sub_group_one_hot[0])

            groups_of_sub_groups = []
            for group_idx in range(N_group):
                groups_of_sub_groups.append([idx for idx in range(N_sub_group) if sub_group_one_hot[idx][group_idx] == 1])
            
            # extract group path
            groups_of_objects = []
            groups_path = []
            
            for group_idx in range(N_group):
                some_idx = groups_of_sub_groups[group_idx][0]
                size_group = len(groups_of_sub_groups[group_idx])
                
                # find bundle position start idx
                start_i = None
                start_pos = None
                for i, pos_i in enumerate(bundled_sub_groups_path[some_idx]):
                    if i == 0:
                        continue
                    overlap_num = 0
                    for other_idx in groups_of_sub_groups[group_idx]:
                        for pos_j in bundled_sub_groups_path[other_idx]:
                            if np.linalg.norm(np.array(pos_i) - np.array(pos_j)) < 1e-5:
                                overlap_num += 1
                                break
                    if overlap_num == size_group:
                        start_i = i
                        start_pos = pos_i
                        break
                
                # find bundle position end idx
                end_i = None
                end_pos = None
                for i in range(len(bundled_sub_groups_path[some_idx]) - 2, -1, -1):
                    pos_i = bundled_sub_groups_path[some_idx][i]
                    overlap_num = 0
                    for other_idx in groups_of_sub_groups[group_idx]:
                        for pos_j in bundled_sub_groups_path[other_idx]:
                            if np.linalg.norm(np.array(pos_i) - np.array(pos_j)) < 1e-5:
                                overlap_num += 1
                                break
                    if overlap_num == size_group:
                        end_i = i
                        end_pos = pos_i
                        break
                
                group_of_objects = []
                for sub_group_idx in groups_of_sub_groups[group_idx]:
                    # print(f'[DEBUG] sub_group_idx = {sub_group_idx}')
                    # print(f'[DEBUG] len of cur_sub_groups = {len(cur_sub_groups)}')
                    group_of_objects.extend(cur_sub_groups[sub_group_idx])
                
                if start_i is None or end_i is None or start_i >= end_i:
                    # print(f'[DEBUG] Group {group_idx} is end')
                    # the group is end, can never bundled
                    for sub_group_idx in groups_of_sub_groups[group_idx]:
                        sub_group = cur_sub_groups[sub_group_idx]
                        for i in sub_group:
                            # push the current sub group path
                            # final_bundled_path_start[i].extend(bundled_sub_groups_path[sub_group_idx])
                            final_bundled_path_mid[i] = bundled_sub_groups_path[sub_group_idx][1 : -1]
                    
                    final_groups.append(group_of_objects)
                    cur_finished_group.append(group_of_objects)
                else:
                    # print(f'[DEBUG] Group {group_idx} is not end')
                    # get average start and end
                    avg_start_pos = np.zeros(2)
                    avg_end_pos = np.zeros(2)
                    for sub_group_idx in groups_of_sub_groups[group_idx]:
                        avg_start_pos += np.array(bundled_sub_groups_path[sub_group_idx][0])
                        avg_end_pos += np.array(bundled_sub_groups_path[sub_group_idx][-1])
                    avg_start_pos /= len(groups_of_sub_groups[group_idx])
                    avg_end_pos /= len(groups_of_sub_groups[group_idx])

                    group_path = [avg_start_pos.tolist()]
                    group_path.extend(bundled_sub_groups_path[some_idx][start_i : end_i + 1])
                    group_path.append(avg_end_pos.tolist())
                
                    groups_path.append(group_path)
                    groups_of_objects.append(group_of_objects)

                    for sub_group_idx in groups_of_sub_groups[group_idx]:
                        sub_group = cur_sub_groups[sub_group_idx]

                        sub_group_start_path = []
                        for pos in bundled_sub_groups_path[sub_group_idx]:
                            if np.linalg.norm(np.array(pos) - np.array(start_pos)) > 1e-5:
                                sub_group_start_path.append(pos)
                            else:
                                break
                        
                        sub_group_end_path = []
                        for i in range(len(bundled_sub_groups_path[sub_group_idx]) - 1, -1, -1):
                            pos = bundled_sub_groups_path[sub_group_idx][i]
                            if np.linalg.norm(np.array(pos) - np.array(end_pos)) > 1e-5:
                                sub_group_end_path.append(pos)
                            else:
                                break
                        
                        # push the path into each object
                        for i in sub_group:
                            final_bundled_path_start[i].extend(sub_group_start_path[1:])
                            final_bundled_path_mid[i] = group_path[1 : -1]
                            final_bundled_path_end[i].extend(sub_group_end_path[1:])

                            # if i == 55:
                            #     print(f'[DEBUG] group_path = {[[round(x, 2) for x in pos] for pos in group_path]}')
            # debug id=55 path, only preserve 2 character after the .
            # print(f'[DEBUG] final_bundled_path_start[55] = {[tuple(round(x, 2) for x in pos) for pos in final_bundled_path_start[55]]}')
            # print(f'[DEBUG] final_bundled_path_mid[55] = {[tuple(round(x, 2) for x in pos) for pos in final_bundled_path_mid[55]]}')
            # print(f'[DEBUG] final_bundled_path_end[55] = {[tuple(round(x, 2) for x in pos) for pos in final_bundled_path_end[55]]}')

            result_path_all_objects = copy.deepcopy(final_bundled_path_start)
            for i in range(self.N):
                result_path_all_objects[i].extend(final_bundled_path_mid[i])
                reverse_end = copy.deepcopy(final_bundled_path_end[i])
                reverse_end.reverse()
                result_path_all_objects[i].extend(reverse_end)

            self.all_stages_result_path_history.append(result_path_all_objects)
            self.all_stages_path_history.append(bundled_sub_groups_path_history)
            self.all_stages_group_history.append(cur_sub_groups + copy.deepcopy(last_finished_group))
            last_finished_group = copy.deepcopy(cur_finished_group)
            # print -1 of group history
            # print(f'[DEBUG] cur_finished_group = {cur_finished_group}')
            # print(f'[DEBUG] cur_history = {self.all_stages_group_history[-1]}')
            # prepare for next stage
            cur_sub_groups = groups_of_objects
            cur_sub_groups_path = groups_path

            if len(groups_path) == 0:
                # all stage are end
                # print(f'All stage are end, cur_stage={cur_stage}')
                break

            # print(f'[DEBUG] groups_path = {groups_path}')
        
        if len(cur_sub_groups) > 0:
            final_groups.extend(cur_sub_groups)
        
        # debug final group len
        # print(f'[DEBUG] final groups = {final_groups}')
        # print(f'[DEBUG] final groups len = {len(final_groups)}')
        
        self.control_points = self.all_stages_result_path_history[-1]
        
        self.sub_groups_per_group = []
        self.amount_groups = len(final_groups)
        self.anime_groups = np.zeros((self.N, self.amount_groups), dtype=int)
        for group_idx, group in enumerate(final_groups):
            # print(f'[DEBUG] group = {group}')
            self.anime_groups[group, group_idx] = 1
            # the stage1 group is sub group
            # please use stage1_groups to get the sub group
            sub_groups_cur_group = []
            # print group
            # print(f'[DEBUG] group = {group}')
            if len(self.all_stages_group_history) >= 2:
                # print(f'[DEBUG] len(self.all_stages_group_history) = {len(self.all_stages_group_history)}')
                for sub_group in self.all_stages_group_history[1]:
                    # print(f'[DEBUG] sub_group = {sub_group}')
                    if sub_group[0] in group:
                        # print(f'[DEBUG] sub_group in group')
                        sub_groups_cur_group.append(sub_group)
            else:
                sub_groups_cur_group = [group]
            self.sub_groups_per_group.append(sub_groups_cur_group)


    def compute_group(self):
        self.compute_control_point()

         # init results
        self.anime_position = np.zeros((self.amount_groups * self.T + 2, self.N, 2))
        self.anime_start = np.zeros((self.N), dtype=np.int32)
        self.anime_end = np.ones((self.N), dtype=np.int32) * (self.T + 1)
        # print(f'Grouping: {self.anime_groups.shape}')

        if self.debug_mode:
            self.make_debug_video()

    def compute_animation(self):
        if self.only_bundle_mode:
            return

        print('Computing Animation...')
        self.anime_position, self.bundled_position, self.anime_start, self.anime_end = Animsolver.anim_compute(self)
        if self.quit_after_packing:
            return
        self.anime_position = np.array(self.anime_position)
        self.bundled_position = np.array(self.bundled_position)
        self.anime_start = np.array(self.anime_start)
        self.anime_end = np.array(self.anime_end)

    def compute_path(self):
        # do nothing
        return

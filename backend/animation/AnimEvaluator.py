import numpy as np
from abc import ABC, abstractmethod
from utils.Animmetric import Animmetric
from matplotlib import pyplot as plt

class AnimMetric(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def __call__(self, anime_position, anime_groups, anime_start, anime_end):
        pass

    def set_debug_name(self, name):
        self.debug_name = name
    
    def debug_curve_per_time(self, data):
        if not hasattr(self, 'debug_name'):
            return
        # use matplot lib to plot curve per time
        plt.figure(figsize=(12, 6))
        plt.title(f'{self.name} of {self.debug_name}')
        plt.xlabel('time')
        plt.ylabel(self.name)
        plt.plot(data)
        plt.savefig(f'./log/image/MetricsPlot/{self.name}_{self.debug_name}.png')
        plt.close()


class OuterOcclusionMetric(AnimMetric):
    def __init__(self, threshold_occlusion = 10 / 1024):
        super().__init__('Outer Occlusion')
        self.threshold_occlusion = threshold_occlusion
    
    def __call__(self, anime_position, anime_groups, anime_start, anime_end):
        # convert anime_position to list[list[list[float]]]
        anime_position = anime_position.tolist()
        # convert anime_groups dtype to int
        anime_groups = np.array(anime_groups, dtype=int).tolist()

        occlusion_per_time = Animmetric.OuterOcclusion(anime_position, anime_groups, [self.threshold_occlusion for _ in range(len(anime_position[0]))], anime_start, anime_end)
        occlusion_per_time = np.array(occlusion_per_time)
        
        self.debug_curve_per_time(occlusion_per_time)

        return np.average(occlusion_per_time)

class WithinGroupOcclusionMetric(AnimMetric):
    def __init__(self, threshold_occlusion = 10 / 1024):
        super().__init__('Within Group Occlusion')
        self.threshold_occlusion = threshold_occlusion
    
    def __call__(self, anime_position, anime_groups, anime_start, anime_end):
        anime_position = anime_position.tolist()
        anime_groups = np.array(anime_groups, dtype=int).tolist()
        occlusion_per_time_group = Animmetric.WithinGroupOcclusion(anime_position, anime_groups, [self.threshold_occlusion for _ in range(len(anime_position[0]))], anime_start, anime_end)
        occlusion_per_time_group = np.array(occlusion_per_time_group)
        occlusion_per_time = np.mean(occlusion_per_time_group, axis=1)
        self.debug_curve_per_time(occlusion_per_time)

        occlusion_per_group = []
        for i in range(len(anime_groups[0])):
            group_start = np.min([start for j, start in enumerate(anime_start) if anime_groups[j][i] == 1])
            group_end = np.max([end for j, end in enumerate(anime_end) if anime_groups[j][i] == 1])
            occlusion_per_group.append(np.mean(occlusion_per_time_group[group_start:group_end, i]))

        return np.average(occlusion_per_group)

class OverallOcclusionMetric(AnimMetric):
    def __init__(self, threshold_occlusion = 10 / 1024):
        super().__init__('Overall Occlusion')
        self.threshold_occlusion = threshold_occlusion
    
    def __call__(self, anime_position, anime_groups, anime_start, anime_end):
        anime_position = anime_position.tolist()
        anime_groups = np.array(anime_groups, dtype=int).tolist()
        occlusion_per_time = Animmetric.OverallOcclusion(anime_position, anime_groups, [self.threshold_occlusion for _ in range(len(anime_position[0]))], anime_start, anime_end)
        occlusion_per_time = np.array(occlusion_per_time)
        self.debug_curve_per_time(occlusion_per_time)
        
        return np.average(occlusion_per_time)

class DispersionMetric(AnimMetric):
    def __init__(self, threshold_occlusion = 10 / 1024):
        super().__init__('Dispersion')
        self.threshold_occlusion = threshold_occlusion
    
    def __call__(self, anime_position, anime_groups, anime_start, anime_end):
        anime_position = anime_position.tolist()
        anime_groups = np.array(anime_groups, dtype=int).tolist()
        dispersion_per_time_group = Animmetric.Dispersion(anime_position, anime_groups, [self.threshold_occlusion for _ in range(len(anime_position[0]))], anime_start, anime_end)
        dispersion_per_time_group = np.array(dispersion_per_time_group)
        dispersion_per_time = np.mean(dispersion_per_time_group, axis=1)
        self.debug_curve_per_time(dispersion_per_time)

        dispersion_per_group = []
        for i in range(len(anime_groups[0])):
            group_start = np.min([start for j, start in enumerate(anime_start) if anime_groups[j][i] == 1])
            group_end = np.max([end for j, end in enumerate(anime_end) if anime_groups[j][i] == 1])
            dispersion_per_group.append(np.mean(dispersion_per_time_group[group_start:group_end, i]))

            print(f'    dispersion of group {i}({group_start}-{group_end}) : {dispersion_per_group[-1]}')
        
        return np.average(dispersion_per_group)

class DeformationMetric(AnimMetric):
    def __init__(self, threshold_occlusion = 10 / 1024):
        super().__init__('Deformation')
        self.threshold_occlusion = threshold_occlusion

    def __call__(self, anime_position, anime_groups, anime_start, anime_end):
        anime_position = anime_position.tolist()
        anime_groups = np.array(anime_groups, dtype=int).tolist()
        deformation_per_time_group = Animmetric.Deformation(anime_position, anime_groups, [self.threshold_occlusion for _ in range(len(anime_position[0]))], anime_start, anime_end)
        deformation_per_time_group = np.array(deformation_per_time_group)
        deformation_per_time = np.mean(deformation_per_time_group, axis=1)
        self.debug_curve_per_time(deformation_per_time)

        deformation_per_group = []
        for i in range(len(anime_groups[0])):
            group_start = np.min([start for j, start in enumerate(anime_start) if anime_groups[j][i] == 1])
            group_end = np.max([end for j, end in enumerate(anime_end) if anime_groups[j][i] == 1])
            deformation_per_group.append(np.mean(deformation_per_time_group[group_start:group_end, i]))

            print(f'    deformation of group {i}({group_start-group_end}) : {deformation_per_group[-1]}')

        return np.average(deformation_per_group)


class ToughnessMetric(AnimMetric):
    def __init__(self):
        super().__init__('Toughness')

    def __call__(self, anime_position, anime_groups):
        T = anime_position.shape[0]
        N = anime_position.shape[1]
        eps = 1e-6

        toughness = np.zeros((N))
        for i in range(N):
            amount_T = 0
            for t in range(2, T):
                dir_1 = anime_position[t - 1, i] - anime_position[t - 2, i]
                dir_2 = anime_position[t, i] - anime_position[t - 1, i]

                dis_1 = np.linalg.norm(dir_1)
                dis_2 = np.linalg.norm(dir_2)
                if dis_1 < eps or dis_2 < eps:
                    continue
                amount_T += 1

                # get angle between dir1 and dir2
                val = np.dot(dir_1, dir_2) / (dis_1 * dis_2)
                if val > 1:
                    val = 1
                    
                angle = np.arccos(val)
                toughness[i] += angle
            toughness[i] /= amount_T
        
        return np.average(toughness)

class TimeMetric(AnimMetric):
    def __init__(self):
        super().__init__('Animation Time')

    def __call__(self, anime_position, anime_groups):
        return anime_position.shape[0]

class GroupNumberMetric(AnimMetric):
    def __init__(self):
        super().__init__('Group Number')

    def __call__(self, anime_position, anime_groups, anime_start, anime_end):
        return anime_groups.shape[1]

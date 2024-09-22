import matplotlib.pyplot as plt
import utils.Bundlemetric.Bundlemetric as Bundlemetric
from animation.AnimData import *
import numpy as np

def plot_paths(path1, path2, distance, ax):
    # 定义两条路径的颜色
    path1_color = 'blue'
    path2_color = 'green'
    
    # 将路径分解为 x 和 y 坐标以便绘制
    path1_x, path1_y = zip(*path1)
    path2_x, path2_y = zip(*path2)
    
    # 绘制路径并指定颜色
    ax.plot(path1_x, path1_y, color=path1_color, label='Path 1')
    ax.plot(path2_x, path2_y, color=path2_color, label='Path 2')
    
    # 标记路径点并指定颜色
    ax.scatter(path1_x, path1_y, color=path1_color)
    ax.scatter(path2_x, path2_y, color=path2_color)
    
    # 添加路径方向箭头并指定颜色
    ax.annotate('', xy=path1[-1], xytext=path1[-2], arrowprops=dict(arrowstyle="->", lw=1, color=path1_color))
    ax.annotate('', xy=path2[-1], xytext=path2[-2], arrowprops=dict(arrowstyle="->", lw=1, color=path2_color))
    
    ax.set(xlim=(0, 1), ylim=(0, 1))

    ax.set_aspect('equal')
    ax.grid(True)
    
    # 设置图例和标题
    ax.legend()
    compa = 1 / (1 + distance)
    ax.set_title(f'DTW:{distance:.2f}, kappa:{compa:.2f}')

if __name__ == '__main__':
    dataset_name = 'ComplexCurve3'
    all_paths = load_data(dataset_name)
    all_paths_n = len(all_paths)
    sample_path_n = 5

    paths = []

    for i in range(sample_path_n):
        path_id1 = np.random.randint(0, all_paths_n)
        path_id2 = np.random.randint(0, all_paths_n)
        while path_id1 == path_id2:
            path_id2 = np.random.randint(0, all_paths_n)
        
        paths.append((all_paths[path_id1], all_paths[path_id2]))

    fig, axs = plt.subplots(5, figsize=(4, 12))
    for i, (path1, path2) in enumerate(paths):
        distance = Bundlemetric.get_DTW(path1, path2)
        plot_paths(path1, path2, distance, axs[i])
    
    plt.tight_layout()
    plt.show()

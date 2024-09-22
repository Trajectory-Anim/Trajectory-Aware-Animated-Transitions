import numpy as np
import matplotlib.pyplot as plt
import cv2
from interface import pd_packing_interface, new_pd_packing_interface


# def generate_test_data(N=6):
#     # random generate N points with start_position and end_position
#     start_position = np.random.rand(N, 2)/2
#     end_position = np.random.rand(N, 2)/2
#     radii = np.random.rand(N) * 0.1 + 0.1
#     sub_clusters = [np.arange(N)]
#     final_pos = pd_packing_interface(start_position, sub_clusters, radii)
#     final_pos = np.array(final_pos)
#     print(f'final_pos = {final_pos}')
#
#     incremental_position = np.random.rand(N//5, 2)
#     incremental_radii = np.random.rand(N//5) * 0.1 + 0.1
#     img = np.zeros((1000, 1000, 3), dtype=np.uint8)
#     for i in range(N):
#         cv2.circle(img, (int(final_pos[i][0]*1000), int(final_pos[i][1]*1000)), int(radii[i]*1000), (255, 255, 255), 1)
#     for i in range(N//5):
#         cv2.circle(img, (int(incremental_position[i][0]*1000), int(incremental_position[i][1]*1000)), int(incremental_radii[i]*1000), (0,0,255), 2)
#     cv2.imwrite('test0.png', img)
#
#
#     # stack the incremental data
#     start_position = np.vstack((start_position, incremental_position))
#     radii = np.hstack((radii, incremental_radii))
#     sub_clusters = [np.arange(N+N//5)]
#     final_pos = pd_packing_interface(start_position, sub_clusters, radii)
#     final_pos = np.array(final_pos)
#     img = np.zeros((1000, 1000, 3), dtype=np.uint8)
#     for i in range(N+N//5):
#         cv2.circle(img, (int(final_pos[i][0]*1000), int(final_pos[i][1]*1000)), int(radii[i]*1000), (255, 255, 255), 2)
#     cv2.imwrite('test.png', img)

def generate_test_data(N=30):
    # random generate N points with start_position and end_position
    # N//3 points in the first cluster, start positions in [0,0]~[0.1,0.1], end positions in [0.9,0.9]~[1,1]
    # N//3 points in the second cluster, start positions in [0.1,0.1]~[0.2,0.2], end positions in [0.8,0.8]~[0.9,0.9]
    # N//3 points in the third cluster, start positions in [0.2,0.2]~[0.3,0.3], end positions in [0.7,0.7]~[0.8,0.8]
    start_positions = np.zeros((N, 2))
    end_positions = np.zeros((N, 2))
    radii = np.random.rand(N) * 0.02 + 0.02
    start_positions[:N//3, :] = np.random.rand(N//3, 2) * 0.1
    start_positions[:N//3, 1] += 0.3
    end_positions[:N//3, :] = np.random.rand(N//3, 2) * 0.1 + 0.9
    end_positions[:N//3, 1] -= 0.3
    start_positions[N//3:2*N//3, :] = np.random.rand(N//3, 2) * 0.1 + 0.1
    start_positions[N//3:2*N//3, 1] += 0.3
    end_positions[N//3:2*N//3, :] = np.random.rand(N//3, 2) * 0.1 + 0.8
    end_positions[N//3:2*N//3, 1] -= 0.3
    start_positions[2*N//3:, :] = np.random.rand(N//3, 2) * 0.1 + 0.2
    start_positions[2*N//3:, 1] += 0.3
    start_positions[2*N//3:, 0] -= 0.2
    end_positions[2*N//3:, :] = np.random.rand(N//3, 2) * 0.1 + 0.7
    end_positions[2*N//3:, 1] -= 0.3
    n_pre = 3
    # pre_packing_pos = start_positions.copy()
    pre_packing_pos = np.zeros((N, 2))
    pre_sub_clusters_id = [np.arange(N//3), np.arange(N//3, 2*N//3), np.arange(2*N//3, N)]
    pre_avg_end_pos = np.zeros((N, 2))
    pre_avg_end_pos[:N//3, :] = np.mean(end_positions[:N//3, :], axis=0)
    pre_avg_end_pos[N//3:2*N//3, :] = np.mean(end_positions[N//3:2*N//3, :], axis=0)
    pre_avg_end_pos[2*N//3:, :] = np.mean(end_positions[2*N//3:, :], axis=0)
    pre_avg_start_pos = np.zeros((N, 2))
    pre_avg_start_pos[:N//3, :] = np.mean(start_positions[:N//3, :], axis=0)
    pre_avg_start_pos[N//3:2*N//3, :] = np.mean(start_positions[N//3:2*N//3, :], axis=0)
    pre_avg_start_pos[2*N//3:, :] = np.mean(start_positions[2*N//3:, :], axis=0)
    pre_radius = radii.copy()
    # for i in range(n_pre):
    #     pre_avg_end_pos[i, :] = np.mean(end_positions[pre_sub_clusters_id[i], :], axis=0)
    #     pre_avg_start_pos[i, :] = np.mean(start_positions[pre_sub_clusters_id[i], :], axis=0)
    init_pos, final_pos = new_pd_packing_interface(n_pre, pre_packing_pos, pre_avg_end_pos, pre_avg_start_pos, pre_radius, pre_sub_clusters_id)
    final_pos = np.array(final_pos)
    final_pos += np.array([0.4, 0.5])
    init_pos = np.array(init_pos)
    init_pos += np.array([0.4, 0.5])
    print(f'init_pos = {init_pos}')
    print(f'final_pos = {final_pos}')
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    for i in range(N):
        cv2.circle(img, (int(final_pos[i][0]*1000), int(final_pos[i][1]*1000)), int(radii[i]*1000), (255, 255, 255), 1)
        cv2.circle(img, (int(start_positions[i][0]*1000), int(start_positions[i][1]*1000)), int(radii[i]*1000), (0, 0, 255), 1)
        cv2.circle(img, (int(end_positions[i][0]*1000), int(end_positions[i][1]*1000)), int(radii[i]*1000), (0, 255, 0), 1)
        cv2.circle(img, (int(init_pos[i][0]*1000), int(init_pos[i][1]*1000)), int(radii[i]*1000), (255, 0, 0), 1)
    cv2.imwrite('test0.png', img)


    # new_pd_packing_interface(n_pre, pre_packing_pos, pre_avg_end_pos, pre_avg_start_pos, pre_radius, pre_sub_clusters_id)




if __name__ == '__main__':
    generate_test_data(9)




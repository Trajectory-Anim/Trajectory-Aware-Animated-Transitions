from sklearn.neighbors import KDTree
import numpy as np
import faiss
import time

# def get_control_points_kNN(control_points, kp):
#     N = len(control_points)
#     id_map = {}
#     id_map_inv = []
#     count = 0

#     # init map
#     for p in range(N):
#         N_cp = len(control_points[p])
#         for i in range(N_cp):
#             id_map[(p, i)] = count
#             id_map_inv.append((p, i))
#             count += 1
    
#     X = np.zeros((count, 2))
#     for i in range(count):
#         X[i, 0] = control_points[id_map_inv[i][0]][id_map_inv[i][1]][0]
#         X[i, 1] = control_points[id_map_inv[i][0]][id_map_inv[i][1]][1]
    
#     tree = KDTree(X)
#     _, idx = tree.query(X, k=int(count * kp))

#     res = []
#     for p in range(N):
#         res_point = []
#         N_cp = len(control_points[p])
#         for i in range(N_cp):
#             res_control_point = []
#             mapped_id = id_map[(p, i)]

#             for kNNid in idx[mapped_id]:
#                 res_control_point.append(id_map_inv[kNNid])
#             res_point.append(res_control_point)
#         res.append(res_point)
    
#     return res


def get_control_points_kNN_faiss(control_points, kNN_k):
    time_start = time.time()

    N = len(control_points)
    id_map = {}
    id_map_inv = []
    count = 0

    # 初始化映射
    for p in range(N):
        N_cp = len(control_points[p])
        for i in range(N_cp):
            id_map[(p, i)] = count
            id_map_inv.append((p, i))
            count += 1
    
    # 准备数据
    X = np.zeros((count, 2), dtype='float32')  # FAISS要求数据为float32
    for i in range(count):
        X[i] = control_points[id_map_inv[i][0]][id_map_inv[i][1]]
    
    # 创建FAISS索引
    index = faiss.IndexFlatL2(2)  # 使用2维空间的L2距离索引

    # 添加数据到索引
    index.add(X)
    
    # 执行kNN搜索
    # k = int(count * kp)  # 确定每个点的邻居数量

    time_search_start = time.time()
    _, idx = index.search(X, kNN_k)
    time_search_end = time.time()

    # 构建结果
    res = []
    for p in range(N):
        res_point = []
        N_cp = len(control_points[p])
        for i in range(N_cp):
            res_control_point = []
            mapped_id = id_map[(p, i)]

            for kNNid in idx[mapped_id]:
                res_control_point.append(id_map_inv[kNNid])
            res_point.append(res_control_point)
        res.append(res_point)
    time_end = time.time()

    print(f'kNN Count : {count}')
    print(f'kNN k = {kNN_k}')
    print(f'kNN search time: {time_search_end - time_search_start}s')
    print(f'kNN time(in Python): {time_end - time_start}s')
    return res

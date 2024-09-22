from sklearn.cluster import AgglomerativeClustering
import numpy as np
import time
import matplotlib.pyplot as plt
import networkx as nx

def get_subgroup_result(distance_matrix, distance_threshold, stage):
    distance_matrix = np.array(distance_matrix)
    
    # 检查样本数量
    if distance_matrix.shape[0] < 2:
        # 如果只有一个样本，直接返回一个单元素的一维数组
        return [[1]]

    symmetric_distance_matrix = np.minimum(distance_matrix, distance_matrix.T)

    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', compute_full_tree=True, linkage='average', distance_threshold=distance_threshold)
    # clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete', distance_threshold=distance_threshold)
    # 使用距离矩阵进行拟合
    clustering.fit(symmetric_distance_matrix)


    if stage > 1:
        labels = clustering.labels_
        # get one hot encoding
        one_hot_encoding = np.zeros((distance_matrix.shape[0], len(np.unique(labels))), dtype=int)
        one_hot_encoding[np.arange(distance_matrix.shape[0]), labels] = 1
        return one_hot_encoding.tolist()

    
    # get tree structure from clustering
    N = distance_matrix.shape[0]

    children = [[None, None] for node in range(N)]
    for node in range(N - 1):
        left = clustering.children_[node][0]
        right = clustering.children_[node][1]
        children.append([left, right])

    capacity = np.zeros((2 * N - 1))
    E_sim = np.zeros((2 * N - 1))
    E_str = np.zeros((2 * N - 1))

    graph = nx.DiGraph()
    source = 'S'
    sink = 'T'

    # get fathers
    fathers = [None] * (2 * N - 1)
    for node in range(N, 2 * N - 1):
        left = children[node][0]
        right = children[node][1]
        fathers[left] = node
        fathers[right] = node
    fathers[2 * N - 2] = source

    # get distance to root for each node
    dis_to_root = [None] * (2 * N - 1)

    def get_distance_to_root(node, distance):
        dis_to_root[node] = distance
        if node >= N:
            left = children[node][0]
            right = children[node][1]
            get_distance_to_root(left, distance + 1)
            get_distance_to_root(right, distance + 1)

    get_distance_to_root(2 * N - 2, 0)

    graph.add_node(source)
    graph.add_node(sink)
    graph.add_nodes_from(range(0, 2 * N - 1))

    # add edges from leaves to sink
    for node in range(N):
        graph.add_edge(node, sink, capacity=float('inf'))
    
    def get_leaves(node):
        if node < N:
            return [node]
        else:
            left = children[node][0]
            right = children[node][1]
            return get_leaves(left) + get_leaves(right)
    
    w_similarity = 1.0
    w_struct = 1.0
    
    def compute_capacity_from_father(node):
        cur_capacity = 0.0

        # compute similarity energy
        cur_E_sim = 0.0
        if node >= N:
            left = children[node][0]
            right = children[node][1]
            left_nodes = get_leaves(left)
            right_nodes = get_leaves(right)
            for left_node in left_nodes:
                for right_node in right_nodes:
                    cur_E_sim += symmetric_distance_matrix[left_node, right_node]
            # E_sim /= len(left_nodes) * len(right_nodes)
        cur_capacity += w_similarity * cur_E_sim

        # compute structure energy
        leaves = get_leaves(node)
        avg_dis_to_leaves = 0.0
        for leaf in leaves:
            avg_dis_to_leaves += dis_to_root[leaf] - \
                dis_to_root[node]
        avg_dis_to_leaves /= len(leaves)
        cur_E_str = dis_to_root[node] / \
            (dis_to_root[node] + avg_dis_to_leaves)
        cur_capacity += w_struct * cur_E_str

        capacity[node] = cur_capacity
        E_sim[node] = cur_E_sim
        E_str[node] = cur_E_str

        # print(f'node: {node}, capacity: {capacity:.3f}, sim: {self.w_similarity * E_sim:.3f}, occ: {self.w_occlusion * E_occ:.3f}, str: {self.w_struct * E_str:.3f}, children: {self.children[node]}')
        return cur_capacity

    for node in range(2 * N - 1):
        father = fathers[node]
        cur_capacity = compute_capacity_from_father(node)
        graph.add_edge(father, node, capacity=cur_capacity)
    
    cut_value, partition = nx.minimum_cut(graph, source, sink)
    reachable, non_reachable = partition

    group_roots = []
    for node in range(2 * N - 1):
        father = fathers[node]
        if father in reachable and node in non_reachable:
            group_roots.append(node)
    
    amount_group = len(group_roots)

    group_one_hot = np.zeros((N, amount_group), dtype=int)

    for cluster_idx, group_root in enumerate(group_roots):
        cur_group = get_leaves(group_root)
        for node in cur_group:
            group_one_hot[node, cluster_idx] = 1
    
    return group_one_hot
    

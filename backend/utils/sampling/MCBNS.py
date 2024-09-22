import random
import this
from unicodedata import category
from markupsafe import re
import numpy as np
from scipy.spatial.distance import cdist
import os   
import cffi
import sys
import time
import math
import functools

from sklearn import neighbors
from utils.thread import FuncThread

import faiss

blue_noise_fail_rate = 0.5

# for efficient add, remove, and random select
# modified from https://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice
class ListDict(object):
    def __init__(self, items = []):
        self.items = items
        self.item_to_position = {}
        for i in range(len(items)):
            self.item_to_position[items[i]] = i
            
    def __len__(self):
        return len(self.items)

    def add_item(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

    def remove_item(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random_item(self):
        return random.choice(self.items)


class MultiClassBlueNoiseSamplingFAISS():
    def __init__(self, dimension, distribution, failure_tolerance=1000, class_weight={}, all_weight=1.0, remove=True, adaptive=False):
        self.failure_tolerance = failure_tolerance
        self.class_weight = class_weight
        self.all_weight = all_weight
        self.remove = remove
        self.adaptive = adaptive
        self.indexer = faiss.IndexFlatL2(dimension)
        self.constraint_matrix = self._build_constraint_matrix(distribution)
        return

    def sample(self, data, category=None):
        neighbors = self._neighbors(self.indexer, data, self.constraint_matrix)
        if self._conflict_check(neighbors.shape[0]):
            self.indexer.add(data)
        return
    def sampling_count(self):
        return self.indexer.ntotal
    def _neighbors(self, indexer, data, constraint_matrix):
        lims, D ,I=indexer.range_search(data, constraint_matrix)
        return I
    def _conflict_check(self, neighbors_count):
        return neighbors_count==0
    def _neighbors_removable(self, idx, selected_idx, neighbors, category, constraint_matrix, cate_fill_rate):
        for neigh in neighbors:
            neigh_idx = selected_idx[neigh]
            if constraint_matrix[category[neigh_idx]][category[neigh_idx]] >= constraint_matrix[category[idx]][category[idx]]:
                return False
            if cate_fill_rate[category[neigh_idx]] < cate_fill_rate[category[idx]]:
                return False
        return True
    def _build_constraint_matrix(self, distribution):
        if distribution=="norm":
            data = np.random.rand(10000,2)
        data = data.astype('float32')
        indexer = faiss.IndexFlatL2(2)
        indexer.add(data)
        dist , _ = indexer.search(data, 50) # why k
        radius = np.average(np.sqrt(dist[:, -1]))
        return float(radius)

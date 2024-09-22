import numpy as np
import os
import csv
import sys
from .SamplingMethods import *


class Sampler(object):
    def __init__(self):
        self.data = None
        self.category = None
        self.sampling_method = None
        self.cache_path = None
        self.dataset_name = None



    def set_data(self, data, category):
        self.data = data
        self.category = category
        self.cache_path = None

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def set_sampling_method(self, method, **kwargs):
        new_kwargs = kwargs.copy()
        self.sampling_method = method(**new_kwargs)
        # if self.if_outlier_score_cached():
        #     new_kwargs['outlier_score'] = self.load_outlier_score_cache()
        #     self.sampling_method = method(**new_kwargs)
        # else:
        #     self.sampling_method = method(**kwargs)

    def get_samples_idx(self):
        np.random.seed(0)
        if self.data is None:
            raise ValueError("Sampler.py: data not exist")
        if self.sampling_method is None:
            raise ValueError("Sampler.py: sampling method not specified")
        selected_indexes = self.sampling_method.sample(self.data, self.category)
        # if not self.if_outlier_score_cached():
        #     self.save_outlier_score_cache(self.sampling_method.outlier_score)
        return selected_indexes

    def get_samples(self):
        selected_indexes = self.get_samples_idx()
        return self.data[selected_indexes], self.category[selected_indexes]

    def if_outlier_score_cached(self):
        if self.cache_path is None:
            # generate the cache path
            dataset_dir = os.path.join(config.data_root, 'cache', self.dataset_name)
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            cache_dir = os.path.join(config.data_root, 'cache', self.dataset_name, 'outlier_score')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            cache_name = self.sampling_method.__str__()+'.npy'
            self.cache_path = os.path.join(cache_dir, cache_name)
            if os.path.exists(self.cache_path):
                return True
            else:
                return False
        else:
            if os.path.exists(self.cache_path):
                return True
            else:
                return False
    def save_outlier_score_cache(self, outlier_score):
        np.save(self.cache_path, outlier_score)

    def load_outlier_score_cache(self):
        return np.load(self.cache_path)
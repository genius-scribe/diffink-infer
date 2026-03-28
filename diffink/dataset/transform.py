import math
import numpy as np


class Transform:
    def __init__(self, data_fixed_length, prob=0.5):
        self.data_fixed_length = data_fixed_length
        self.prob = prob

    def random_scaling(self, data, scale_range=(0.9, 1.1)):
        scale = np.random.uniform(*scale_range)
        data[:, :2] *= scale
        return data

    def random_rotation(self, data, angle_range=(-math.pi / 36, math.pi / 36)):
        angle = np.random.uniform(*angle_range)
        rotation_matrix = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        data[:, :2] = np.dot(data[:, :2], rotation_matrix)
        return data

    def augment_data(self, data):
        augmented = data.copy()
        for method in [self.random_scaling, self.random_rotation]:
            if np.random.rand() < self.prob:
                augmented = method(augmented)
        return augmented

    def __call__(self, data):
        return self.augment_data(data)

"""
从mat文件中读取数据
"""

import scipy.io as sio
import numpy as np

def parseMat(mat_path):
    """
    从mat文件中读取数据
    :param mat_path: mat文件路径
    :return: 数据
    """
    data = sio.loadmat(mat_path)
    result = {}
    for key in data.keys():
        if '__' not in key:
            result[key] = data[key]
    return result

class VoxelData:
    def __init__(self, mat_path):
        self.raw = parseMat(mat_path)
        print(self.raw.keys())
        self.volumn = self.raw['volumn']
        self.volumn_vertices = self.raw['volumn_vertices']
        self.volumn_samples = self.raw['volumn_samples']
        self.faces = self.raw['faces']
        self.rotate_axisangle = self.raw['rotate_axisangle']
        self.voxel_centers = self.raw['voxel_centers']


if __name__ == '__main__':
    mat_path = './datasets/train/1a40eaf5919b1b3f3eaa2b95b99dae6_r1.mat'
    data = VoxelData(mat_path)
    print(data.volumn.shape)
    print(data.volumn_vertices.shape)
    print(data.volumn_samples.shape)
    print(data.faces.shape)
    print(data.rotate_axisangle.shape)
    print(data.voxel_centers.shape)
        
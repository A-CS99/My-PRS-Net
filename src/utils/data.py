"""
从mat文件中读取数据
"""
import scipy.io as sio
import numpy as np
import torch
import os
from torch.utils.data import Dataset

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

class RawData:
    '''
    原始数据类
    '''
    def __init__(self, data):
        self.data = data
        self.volumn = torch.tensor(data['volumn'], dtype=torch.float32)
        self.vertices = torch.tensor(data['volumn_vertices'], dtype=torch.float32)
        self.samples = torch.tensor(data['volumn_samples'], dtype=torch.float32)
        self.faces = torch.tensor(data['faces'], dtype=torch.float32)
        self.rotate_axisangle = torch.tensor(data['rotate_axisangle'], dtype=torch.float32)
        self.voxel_centers = torch.tensor(data['voxel_centers'], dtype=torch.float32)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        pass

    def volumn(self):
        return self.volumn
    
    def vertices(self):
        return self.vertices
    
    def samples(self):
        return self.samples
    
    def faces(self):
        return self.faces
    
    def rotate_axisangle(self):
        return self.rotate_axisangle
    
    def voxel_centers(self):
        return self.voxel_centers

class ShapeNetData:
    """
    基于ShapeNet数据集的数据加载类
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dir_list = os.listdir(self.root_dir)
    
    def __len__(self):
        return len(self.dir_list)
    
    def __getitem__(self, idx):
        mat_path = os.path.join(self.root_dir, self.dir_list[idx])
        data = parseMat(mat_path)
        return data

class ShapeNetDataset(Dataset):
    '''
    体素模型的数据加载类
    '''
    def __init__(self, root_dir):
        self.data = ShapeNetData(root_dir)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        volumn = torch.tensor(data['volumn'], dtype=torch.float32)
        volumn = volumn.expand(1, 32, 32, 32)
        return volumn
    
    def raw_data(self, idx):
        raw_data = self.data[idx]
        return RawData(raw_data)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    root_dir = './datasets/train/'
    dataset = ShapeNetDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for i, data in enumerate(dataloader):
        print(data.shape)
        break
        
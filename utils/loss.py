"""
实现两种损失函数
    1. 对称距离损失函数
    2. 正则化损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def symmetry_points(points: torch.Tensor, plane: torch.Tensor):
    '''
    对称化点云
    @param points: 点云 3xN
    @param plane: 对称平面 1x4
    @return sym_points: 对称化点云 3xN
    '''
    # 1. 截取对称平面的单位法向量 1x3
    plane_normal = plane[:3]
    d_value = plane[3]
    L2_norm = torch.norm(plane_normal)
    unit_normal = plane_normal / L2_norm
    # 2. 计算对称面到点的距离 1xN
    dist2plane = torch.sub(torch.matmul(unit_normal, points), d_value)
    # 3. 计算差值向量（两倍的距离）
    t_unit_normal = torch.unsqueeze(unit_normal, dim=1)
    sub_vector = torch.mul(t_unit_normal, dist2plane)
    # 4. 对称化点云
    sym_points = torch.sub(points, sub_vector, alpha=2)
    return sym_points

def rotate_points(points: torch.Tensor, quaternion: torch.Tensor):
    '''
    旋转点云
    @param points: 点云 3xN
    @param quaternion: 旋转四元数 1x4
    @return rotated_points: 旋转后的点云 3xN
    '''
    # 1. 计算旋转矩阵
    w, x, y, z = quaternion
    rotation_matrix = quat2Rmatrix(quaternion)
    # 2. 点云旋转
    rotated_points = torch.matmul(rotation_matrix, points)
    return rotated_points

def quat2Rmatrix(quaternion: torch.Tensor):
    '''
    将四元数转换为旋转矩阵
    @param quaternion: 旋转四元数 1x4
    @return rotation_matrix: 旋转矩阵 3x3
    '''
    w, x, y, z = quaternion
    # rotation_matrix = torch.tensor([
    #     [1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],
    #     [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],
    #     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]
    # ])
    rotation_matrix = torch.stack([
        torch.stack([1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y]),
        torch.stack([2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x]),
        torch.stack([2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2])
    ])
    return rotation_matrix
    
def shortest_distance(points: torch.Tensor, sample: torch.Tensor):
    '''
    计算点到点云的最短距离
    @param points: 点云 3xN
    @param sample: 采样点 3x1
    @return distance: 最短距离 1xM
    '''
    # 1. 计算所有样本点到点云间的距离
    all_distance = torch.norm(torch.sub(points, torch.unsqueeze(sample, dim=1)), dim=0)
    # 2. 计算最短距离
    shortest_distance = torch.min(all_distance)
    return shortest_distance

def SDELoss(planes: torch.Tensor, quaternions: torch.Tensor, samples: torch.Tensor, points: torch.Tensor):
    '''
    对称距离损失函数
    @param planes: 预测的对称平面 3x4
    @param quaternions: 预测的旋转轴 3x4
    @param samples: 采样点 3xM
    @param points: 模型所有顶点 3xN
    @return loss: 对称距离损失
    '''
    # 一、计算对称平面的对称距离损失SDE
    plane_loss = 0
    for plane in planes:
        # 1. 计算对称后的采样点
        sym_samples = symmetry_points(samples, plane)
        # 2. 计算采样点到形状的最短距离
        for sample in sym_samples.transpose(0, 1):
            dist = shortest_distance(points, sample)
            # 3. 累加损失
            plane_loss += dist
    
    # 二、计算旋转轴的对称距离损失SDE
    quaternion_loss = 0
    for quat in quaternions:
        # 1. 计算旋转后的采样点
        rotated_samples = rotate_points(samples, quat)
        # 2. 计算采样点到形状的最短距离
        for sample in rotated_samples.transpose(0, 1):
            dist = shortest_distance(points, sample)
            # 3. 累加损失
            quaternion_loss += dist
    
    # 三、计算总损失
    sde_loss = plane_loss + quaternion_loss
    return sde_loss

def RegLoss(planes, quaternions):
    # 正则化损失函数
    # 计算两个旋转轴之间的距离
    # planes: 预测的对称平面 3x4
    # quaternions: 预测的旋转轴 3x4
    # 返回正则化损失

    return loss

def SumLoss(planes, quaternions, samples, points, reg_weight=25):
    '''
    总损失函数
    @param planes: 预测的对称平面 3x4
    @param quaternions: 预测的旋转轴 3x4
    @param samples: 采样点 3xM
    @param points: 模型所有顶点 3xN
    @param weight: 正则化损失的权重
    @return sum_loss: 总损失
    '''
    sde_loss = SDELoss(planes, quaternions, samples, points)
    reg_loss = RegLoss(planes, quaternions)
    sum_loss = sde_loss + reg_weight * reg_loss
    return sum_loss

if __name__ == '__main__':
    # 测试SDELoss
    planes = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32, requires_grad=True)
    quaternions = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.866, 0.866, 0.866], [1, 0, 0, 0]], dtype=torch.float32, requires_grad=True)
    # 完全非对称的点云
    samples = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=torch.float32)
    # 平面对称的点云
    # points = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=torch.float32)
    # 平面对称且旋转对称的点云
    # points = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, -1, 1], [2, 2, 2, 2, -2]], dtype=torch.float32)
    points = torch.tensor([[0, 0, 0, 0, 0, 2], [1, 1, 1, -1, 1, 0], [2, 2, 2, 2, -2, 1]], dtype=torch.float32)
    sde_loss = SDELoss(planes, quaternions, samples, points)
    print(sde_loss)
    sde_loss.backward()
    print(planes.grad)
    print(quaternions.grad)
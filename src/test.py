import yaml
import torch
from torch.utils.data import DataLoader
from utils.data import ShapeNetDataset
from utils.loss import SDELoss

def test():
    # 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('testing on', device)
    with open('./src/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_path = config['basic']['dataset_path'] + 'test/'
    save_path = config['basic']['save_path']
    sde_bound = config['test']['sde_bound']
    dihedral_angle_bound = config['test']['dihedral_angle_bound']
    
    # 加载模型
    net = torch.load(save_path + 'PRS-Net.pth').to(device)
    net.eval()

    # 加载数据
    dataset = ShapeNetDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 测试
    for i, data in enumerate(dataloader):
        # 只测试一个batch
        if i == 1:
            break
        inputs = data.to(device)
        planes, quaternions = net(inputs)
        samples = dataset.raw_data(i).samples.to(device)
        vertices = dataset.raw_data(i).vertices.to(device)
        planes, quaternions = check_invalid(planes, quaternions, samples, vertices, bound=sde_bound, device=device)
        planes, quaternions = check_repeat(planes, quaternions, samples, vertices, bound=dihedral_angle_bound, device=device)
        print('planes:', planes, sep='\n')
        print('quaternions:', quaternions, sep='\n')
        # 计算损失
        sde_loss = SDELoss(planes, quaternions, samples, vertices, device=device)
        print('sde_loss:', sde_loss)
        print('-----------------------------------')

def check_invalid(planes, quaternions, samples, points, *, bound, device):
    '''
    检查无效的对称平面和旋转轴
    @param planes: 对称平面列表 3x4
    @param quaternions: 旋转轴列表 3x4
    @param samples: 采样点 3xM
    @param points: 模型所有顶点 3xN
    @param device: 设备
    @return planes: 有效的对称平面列表
    @return quaternions: 有效的旋转轴列表
    '''
    # 去除SDE大于阈值的对称平面和旋转轴
    for i in range(len(planes)):
        if plane_sde_loss(planes[i], samples, points, device=device) > bound:
            planes.pop(i)
    for i in range(len(quaternions)):
        if quat_sde_loss(quaternions[i], samples, points, device=device) > bound:
            quaternions.pop(i)
    return planes, quaternions

def check_repeat(planes, quaternions, samples, points, *, bound, device):
    '''
    检查重复的对称平面和旋转轴
    @param planes: 对称平面列表 3x4
    @param quaternions: 旋转轴列表 3x4
    @param device: 设备
    @return planes: 去重后的对称平面列表
    @return quaternions: 去重后的旋转轴列表
    '''
    # 两个法向量的夹角小于阈值则认为是同一个平面，去除SDE较大的平面
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            if dihedral_angle(planes[i], planes[j]) < bound:
                if (plane_sde_loss(planes[i], samples, points, device=device) 
                    < plane_sde_loss(planes[j], samples, points, device=device)):
                    planes.pop(j)
                else:
                    planes.pop(i)
    # 两个旋转轴的夹角小于阈值则认为是同一个旋转轴，去除SDE较大的旋转轴
    for i in range(len(quaternions)):
        for j in range(i + 1, len(quaternions)):
            if dihedral_angle(quaternions[i], quaternions[j]) < bound:
                if (quat_sde_loss(quaternions[i], samples, points, device=device) 
                    < quat_sde_loss(quaternions[j], samples, points, device=device)):
                    quaternions.pop(j)
                else:
                    quaternions.pop(i)
    return planes, quaternions

def dihedral_angle(v1, v2):
    '''
    计算两个向量的二面角
    @param v1: 向量1 1x3
    @param v2: 向量2 1x3
    @return angle: 二面角
    '''
    angle = torch.acos(torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2)))
    return angle

if __name__ == '__main__':
    test()
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.loss import SumLoss
from utils.network import PRS_Net
from utils.data import ShapeNetDataset

def train():
    # 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('training on', device)
    with open('./src/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_path = config['basic']['dataset_path'] + 'train/'
    save_path = config['basic']['save_path']
    # 配置超参数
    hyperparameters = config['hyperparameters']
    epochs = hyperparameters['epochs']
    batch_size = hyperparameters['batch_size']
    learning_rate = hyperparameters['learning_rate']
    reg_wight = hyperparameters['reg_weight']


    # 加载数据
    dataset = ShapeNetDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 加载网络
    net = PRS_Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # 训练
    for epoch in range(epochs):
        print('epoch: ', epoch + 1)
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # 只训练一个batch
            if i == 1:
                break
            print('batch: ', i + 1)
            inputs = data.to(device)
            optimizer.zero_grad()
            planes_batch, quaternions_batch = net(inputs)

            for batch in range(batch_size):
                planes = planes_batch[batch].clone().detach().requires_grad_(True)
                quaternions = quaternions_batch[batch].clone().detach().requires_grad_(True)
                model_idx = i * batch_size + batch
                samples = dataset.raw_data(model_idx).samples.to(device)
                vertices = dataset.raw_data(model_idx).vertices.to(device)
                loss = SumLoss(planes, quaternions, samples, vertices, reg_wight, device=device)
                loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print('loss: ', loss.item())
            if i % 200 == 199:
                print('%d-[%5d, %5d] avg_loss: %.3f' % (epoch + 1, i - 198, i + 1, running_loss / 200))
                running_loss = 0.0
            print('-----------------------------------')
    print('Finished Training')
    torch.save(net.state_dict(), save_path + 'PRS-Net.pth')

if __name__ == '__main__':
    train()
import torchvision
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# 定义使用设备
device = torch.device("cuda")


# 搭建神经网络
class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, 1, padding=2),  # 这个padding要打出来不然会变成其他的属性值
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# 准备数据集

train_data = torchvision.datasets.CIFAR10("./dataset", True, torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", False, torchvision.transforms.ToTensor(), download=True)

# lenth长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

# 创建网络模型
test = Test()
test = test.to(device)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
# 优化器
learning_rate = 0.001
optimizer = torch.optim.SGD(test.parameters(), lr=learning_rate)
# 设置训练网络的一些参数
total_train_step = 0
# 设置训练的次数
total_test_step = 0
# 设置测试的次数
epoch = 200

# 添加tensorboard
writer = SummaryWriter("./logs_train")

start_time = time.time()
for i in range(epoch):
    print("-------从{}开始--------".format(i + 1))
    for data in train_data_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = test(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            # print(end_time-start_time)
            print("训练次数:{},loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    # 测试步骤开始
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = test(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step = total_test_step + 1
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    # 保存每一轮的数据
    torch.save(test, "./testpth/test_{}.pth".format(i))
    # torch.save(test.state.dirt(),"test_{}.pth".format(i))官方推荐保存方式
    print("模型已保存")

writer.close()

# 准备数据集
# 准备dataloader
# 创建网络模型
# 创建损失函数
# 创建优化器
# 创建训练参数
# 创建epoch进行多次训练
# 调用网络模型.train()进入训练状态
# 开始训练
# 进入优化器中优化
# 展示输出
# 测试开始
# 设置with torch.no_grad()表示不需要梯度调整、优化
# 从测试集中提取数据
# 计算loss误差
# 表示正确率
# 保存模型

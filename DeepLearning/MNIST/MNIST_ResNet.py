import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# 检查CUDA是否可用
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=3),  # 将单通道图像转为3通道
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 创建单通道ResNet-18模型
class SingleChannelResNet(nn.Module):
    def __init__(self):
        super(SingleChannelResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # 加载预训练的ResNet-18模型
        # 修改第一层卷积的输入通道数为1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 添加一个新的全连接层作为分类器
        self.fc = nn.Linear(512, 10)  # 假设num_classes为分类的类别数

    def forward(self, x):
        return self.resnet(x)

# 使用单通道预训练的ResNet模型
model = SingleChannelResNet()

# 将模型移动到GPU
model = model.to(DEVICE)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Epoch [{}/{}], Accuracy: {:.2f}%'.format(epoch+1, num_epochs, 100 * correct / total))

import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
# import matplotlib.pyplot as plt


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_to_idx = self._build_label_to_idx()
        self._load_images()
    
    def _build_label_to_idx(self):
        # 假设所有的标签都在self.root_dir下的子目录中
        labels = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        return {label: idx for idx, label in enumerate(labels)}

    def _load_images(self):
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, file)
                    self.images.append(img_path)
                    self.labels.append(self.label_to_idx[label])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)  
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# 定义图像变换
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.456,), (0.224,))
    # ... 其他变换 ...
])
# 创建训练数据集
dataset = ImageFolderDataset(root_dir='/root/FaceEmotion/face_emotion_dataset/train', transform=transform)

# 创建训练数据加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 创建测试数据集
test_dataset = ImageFolderDataset(root_dir='/root/FaceEmotion/face_emotion_dataset/test1', transform=transform)

# 创建测试数据加载器
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7
# 指定模型文件的路径
resnet18_model_path = '/root/PublicModels/resnet18.pth'

# 创建单通道ResNet-18模型
class SingleChannelResNet(nn.Module):
    def __init__(self):
        super(SingleChannelResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)  
        # 加载预训练的ResNet-18模型
        self.resnet.load_state_dict(torch.load(resnet18_model_path))
        # 修改第一层卷积的输入通道数为1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 添加一个新的全连接层作为分类器
        self.fc = nn.Linear(512, num_classes)  # 假设num_classes为分类的类别数

    def forward(self, x):
        return self.resnet(x)

# 使用单通道预训练的ResNet模型
resnet18 = SingleChannelResNet()

resnet18 = resnet18.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(list(resnet18.parameters()) , lr=0.001, momentum=0.9)
num_epochs = 10

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    train_correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # 使用VGG19进行训练
        outputs_resnet18 = model(images)
        loss_resnet18 = criterion(outputs_resnet18, labels)
        optimizer.zero_grad()
        loss_resnet18.backward()
        optimizer.step()
        train_loss += loss_resnet18.item()
        _, predicted = torch.max(outputs_resnet18.data, 1)
        total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        if(i+1)%100 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(images), len(train_loader.dataset),
                100. * i / len(train_loader), loss_resnet18.item()))

    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_correct / total)
            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss_resnet18 = criterion(output, labels)
            test_loss += loss_resnet18.item() # 将一批的损失相加
            _, predicted = torch.max(output.data, 1) # 找到概率最大的下标
            total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(test_correct / total)
 
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, len(test_loader.dataset),
        100. * test_correct / len(test_loader.dataset)))

for epoch in range(1, num_epochs + 1):
    train(resnet18, device, dataloader, optimizer, epoch)
    test(resnet18, device, test_dataloader)
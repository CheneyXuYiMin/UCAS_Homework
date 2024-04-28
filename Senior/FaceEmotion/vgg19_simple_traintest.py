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
vgg19_model_path = '/root/PublicModels/vgg19.pth'

# 创建单通道vgg-19模型
class SingleChannelVgg(nn.Module):
    def __init__(self):
        super(SingleChannelVgg, self).__init__()
        self.vgg = models.vgg19(pretrained=False)  
        # 加载预训练的vgg-19模型
        self.vgg.load_state_dict(torch.load(vgg19_model_path))
        # 修改输入通道数为1
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 修改分类器
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)

# 使用单通道预训练的vgg模型
vgg19 = SingleChannelVgg()

vgg19 = vgg19.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(list(vgg19.parameters()) , lr=0.001, momentum=0.9)
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
        outputs_vgg19 = model(images)
        loss_vgg19 = criterion(outputs_vgg19, labels)
        optimizer.zero_grad()
        loss_vgg19.backward()
        optimizer.step()
        train_loss += loss_vgg19.item()
        _, predicted = torch.max(outputs_vgg19.data, 1)
        total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        if(i+1)%100 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(images), len(train_loader.dataset),
                100. * i / len(train_loader), loss_vgg19.item()))

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
            loss_vgg19 = criterion(output, labels)
            test_loss += loss_vgg19.item() # 将一批的损失相加
            _, predicted = torch.max(output.data, 1) # 找到概率最大的下标
            total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(test_correct / total)
 
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, len(test_loader.dataset),
        100. * test_correct / len(test_loader.dataset)))

for epoch in range(1, num_epochs + 1):
    train(vgg19, device, dataloader, optimizer, epoch)
    test(vgg19, device, test_dataloader)
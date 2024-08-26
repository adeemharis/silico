import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32*128*128, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MultilabelResNet(nn.Module):
    def __init__(self, num_classes):
        super(MultilabelResNet, self).__init__()
        self.resnet = models.resnet50(weights=None)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

        # Load the pretrained weights
        pretrained_dict = torch.load('models/resnet50-0676ba61.pth')
        model_dict = self.resnet.state_dict()
        # state_dict = torch.load('models/resnet50-0676ba61.pth')
        # self.resnet.load_state_dict(state_dict)

        # Filter out the weights that don't match
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)
    
    def forward(self, x):
        return self.resnet(x)

def getModel(num_classes):
    #model = SimpleCNN()
    model = MultilabelResNet(num_classes=num_classes)
    return model
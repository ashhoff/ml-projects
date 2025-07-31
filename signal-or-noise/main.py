import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as tv_models

# where the real and fake dog pictures live
data_dir = "data"
batch_size = 32
img_size = 224

# resize every picture and turn it into numbers the model can understand
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# label folders: real_dogs = 0, fake_dogs = 1
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# send 32 pictures at a time, shuffled
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# use a model that already knows how to rocognize patterns in pictures
dog_judge = tv_models.mobilenet_v2(pretrained=True)

# freeze everything it already knows, just adjusting the last layer
for layer in dog_judge.parameters():
    layer.requires_grad = False

# swap the final layer so it chooses between real and fake
dog_judge.classifier[1] = nn.Linear(dog_judge.last_channel, 2)

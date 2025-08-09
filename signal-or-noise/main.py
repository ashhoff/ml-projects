import os

# This imports datasets and transforms from torchvision. datasets loads images from folders and auto-labels them based on the folder name. transforms resizes and converts each image into a PyTorch tensor, the numerical format the model can process.
from torchvision import datasets, transforms

# This imports the PyTorch DataLoader, which takes an existing dataset and serves it to the model in manageable batches, optionally shuffling the data to improve learning.”
from torch.utils.data import DataLoader

# PyTorch (torch) is the core ML framework. torchvision is an optional add-on for computer vision tasks.
import torch

# torch.nn is PyTorch’s neural network toolbox, basically the LEGO kit for building models. I’m importing it as nn so I can grab layers and loss functions without typing torch.nn every time. It’s where all the core building blocks for creating and training neural networks live.
import torch.nn as nn

# torchvision.models is basically the model zoo in the torchvision library — a grab bag of pre-trained computer vision models like MobileNet, ResNet, and EfficientNet. I’m importing it as tv_models so I can quickly grab one, tweak it, and use it without writing it from scratch.
import torchvision.models as tv_models

# data_dir" points to the folder that holds all my images. "batch_size" is how many images the DataLoader will feed the model at once so it doesn’t overload memory. "img_size" is the size I’m resizing all images to 224×224 pixels because that’s the input size MobileNetV2 is designed to work with.
data_dir = "data"
batch_size = 32
img_size = 224

# transform is just the prep list for my images. Step one: resize everything to 224×224 so it fits what MobileNetV2 expects. Step two: turn the image into a PyTorch tensor, basically a multi-dimensional spreadsheet for fast math and scale pixel values from 0–255 down to 0–1 so training doesn’t get messy.
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# mageFolder is like the clerk in the filing room. You tell them where the cabinet is (root), they go into each folder, slap a numeric tag on it, and make sure each photo inside gets resized and converted to tensors before it’s filed away.
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# dataloader is the variable holding our DataLoader tool. DataLoader takes the dataset we just made and feeds it to the model in batches (so we don’t torch our memory). batch_size is set to the number we chose earlier, and shuffle=True means it randomizes the order of the images each time so the model doesn’t learn patterns based on the order they’re stored in.
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# dog_judge is the variable holding our MobileNetV2 model, which we’re grabbing from tv_models (the torchvision models toolkit). Setting pretrained=True means we’re loading a version that’s already been trained on ImageNet so it already knows how to recognize a ton of visual patterns before we even start.
dog_judge = tv_models.mobilenet_v2(pretrained=True)

# This loop walks through every knob (parameter) in the MobileNetV2 model and tapes them down so they can’t change during training. We do this to keep all the pretrained visual pattern knowledge intact, and only let the new last layer learn how to separate 'real dog' from 'fake dog.'
for layer in dog_judge.parameters():
    layer.requires_grad = False

# We replace the final decision layer in MobileNetV2 with our own Linear layer that takes in the 1,280 features from the previous step (last_channel) and outputs just two classes: real dog or fake dog. This keeps all the earlier visual recognition skills intact but retrains the model to focus only on our specific task.
dog_judge.classifier[1] = nn.Linear(dog_judge.last_channel, 2)

# Make a scoring system that tells us how far off our guesses are from the correct answer, using the Cross Entropy method, which is great for classification problems like real vs. fake dogs.
criterion = nn.CrossEntropyLoss()

# This line creates our optimizer. The algorithm that updates model parameters during training. We’re using Adam, which adapts the learning rate for each parameter automatically, and we’re only giving it the parameters of the new final layer since the rest of the model is frozen. The learning rate is set to 0.001 for small, precise updates instead of large, risky jumps.
optimizer = torch.optim.Adam(dog_judge.classifier[1].parameters(), lr=0.001)

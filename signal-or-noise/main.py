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

# This line creates our optimizer, the algorithm that updates model parameters during training. We’re using Adam, which adapts the learning rate for each parameter automatically, and we’re only giving it the parameters of the new final layer since the rest of the model is frozen. The learning rate is set to 0.001 for small, precise updates instead of large, risky jumps.
optimizer = torch.optim.Adam(dog_judge.classifier[1].parameters(), lr=0.001)

# This line sets the computation device for PyTorch. It checks if a CUDA-enabled GPU is available and, if so, uses it for faster training. Otherwise, it defaults to the CPU. This ensures the code runs on any machine without modification.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dog_judge.to(device)  # model and data must live on the same device

num_epochs = 5  # Number of times the model will see the entire dataset
# This loop will run the training process for 5 epochs, meaning the model will see the entire dataset 5 times. Repeating the process improves learning by allowing the model to adjust its parameters multiple times.
for epoch in range(5):

    # We call .train() on the model to put it into training mode. This ensures that layers like Dropout and BatchNorm behave correctly for learning, rather than acting as they would during inference.
    dog_judge.train()
    epoch_loss = 0.0   # reset running loss counter for this epoch

    # This loop pulls batches of images and their labels from the DataLoader. images contains the batch of resized, tensor-formatted pictures, and labels contains the numeric class tags that correspond to each image’s folder.
    for images, labels in dataloader:

        # .to(device) moves the image and label tensors onto the same compute device (GPU if available, otherwise CPU) as the model. Model and data must be co-located for PyTorch ops to work; this line ensures that for every batch.
        images, labels = images.to(device), labels.to(device)
        # optimizer.zero_grad() wipes the Etch A Sketch before we draw again. When the model learns, it keeps track of how wrong it was using gradients, little direction markers telling it how to tweak its knobs. If we didn’t clear them out each time, those old directions would stack with the new ones and send the model totally off course. This line basically says, “Forget what you learned from the last batch, we’re starting fresh for this one
        optimizer.zero_grad()
        # Run the current batch of images through the model to get raw prediction scores (logits) for each class
        outputs = dog_judge(images)
        # Compare the model’s predictions (outputs) with the actual labels to calculate how wrong it was for this batch
        loss = criterion(outputs, labels)
        # Send the error signal backward through the network so each trainable weight knows how it should adjust
        loss.backward()
        # Reads the gradient “notes” from loss.backward() and actually updates the model’s knobs (parameters) accordingly, this is the step where learning happens.
        optimizer.step()
        # Add the current batch’s loss to the running total for this epoch. This lets us track how much “wrongness” the model accumulates over all batches, so we can later see the average loss for the epoch.
        epoch_loss += loss.item()

    # Calculate the average loss for this epoch by dividing the total loss by the number of batches.
    # If there were no batches (edge case), use 0.0 to avoid division errors.
    avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    # Prints a nice progress report for this epoch. The {epoch+1} and {num_epochs} show which round we’re on out of the total. 
    # {avg_loss:.4f} shows the average loss for the epoch, rounded to 4 decimal places so it looks clean.
    # The f at the start makes this an f-string, letting us drop variables straight into the text like a Mad Lib.
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

# Make a 'models' folder if it doesn’t already exist, then save all the model’s trained knobs (weights) to a file so we can load it later without retraining
os.makedirs("models", exist_ok=True)
torch.save(dog_judge.state_dict(), "models/dog_judge_mobilenetv2.pt")
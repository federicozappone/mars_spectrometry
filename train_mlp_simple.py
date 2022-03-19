import cv2
import time
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pickle

from torch.optim import lr_scheduler

from model import Mars_Spectrometry_Model
from dataset import Mars_Spectrometry_Dataset


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(1600, 512),
      nn.ReLU(),
      nn.Linear(512, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 10)
    )


  def forward(self, x):
    return self.layers(x)
  
  
def train():
    seed_torch(1)

    # initialize custom happywhale dataset and transformations
    dataset_train = Mars_Spectrometry_Dataset("dataset/train_features.pickle", "dataset/train_labels.pickle")
    dataset_val = Mars_Spectrometry_Dataset("dataset/train_features.pickle", "dataset/train_labels.pickle")

    # number of validation samples (30% of training samples)
    num_val_samples = int(len(dataset_train) * 0.3)

    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-num_val_samples])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[-num_val_samples:])

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=4, shuffle=True, num_workers=4)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=4, shuffle=False, num_workers=4)

    # set up dataloaders dict
    dataloaders = {
        "train": data_loader_train,
        "val": data_loader_val
    }

    # select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Mars_Spectrometry_Model(num_features=1600, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    # observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

  
    # Initialize the MLP
    mlp = MLP()
      
    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
      
    # Run the training loop
    for epoch in range(0, 25): # 5 epochs at maximum
        
        # Print epoch
        print(f'Starting epoch {epoch+1}')
        
        # Set current loss value
        current_loss = 0.0
        
        # Iterate over the DataLoader for training data
        for i, data in enumerate(data_loader_train, 0):
          
          # Get inputs
          inputs, targets = data
          
          # Zero the gradients
          optimizer.zero_grad()
          
          # Perform forward pass
          outputs = mlp(inputs)
          
          # Compute loss
          loss = loss_function(outputs, targets)
          
          # Perform backward pass
          loss.backward()
          
          # Perform optimization
          optimizer.step()
          
          # Print statistics
          current_loss += loss.item()

          if i % 700 == 0:
              print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
              current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')


if __name__ == "__main__":
    train()

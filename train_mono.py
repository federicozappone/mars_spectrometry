import numpy as np
import time
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd

from torch.optim import lr_scheduler

from model import Mars_Spectrometry_Model
from dataset import Mars_Spectrometry_Dataset_Mono


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_model(device, model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0

    dataset_sizes = {x: len(dataloaders[x]) for x in ["train", "val"]}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # set model to training mode
            else:
                model.eval()   # set model to evaluate mode

            y_true = []
            y_pred = []

            running_loss = 0.0
            running_corrects = 0

            # iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    loss = criterion(outputs, labels.unsqueeze(0))

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                preds = torch.round(torch.sigmoid(outputs))

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))

            if phase == "train" and scheduler is not None:
                scheduler.step()

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_acc, best_loss
  
  
def train():
    seed_torch(42)

    compounds = "basalt", "carbonate", "chloride", "iron_oxide", "oxalate", "oxychlorine", "phyllosilicate", "silicate", "sulfate", "sulfide"
    compounds_loss = {}

    # select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)

    for compound_index, compound in enumerate(compounds):
        print(f"Training for compound {compound}")

        # initialize custom dataset and transformations
        dataset_train = Mars_Spectrometry_Dataset_Mono("dataset/train_features.pickle", "dataset/train_labels.pickle", compound_index)
        dataset_val = Mars_Spectrometry_Dataset_Mono("dataset/val_features.pickle", "dataset/val_labels.pickle", compound_index)

        """
        # number of validation samples (30% of training samples)
        num_val_samples = int(len(dataset_train) * 0.3)

        indices = torch.randperm(len(dataset_train)).tolist()
        dataset_train = torch.utils.data.Subset(dataset_train, indices[:-num_val_samples])
        dataset_val = torch.utils.data.Subset(dataset_val, indices[-num_val_samples:])
        """

        # define training and validation data loaders
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=1, shuffle=True, num_workers=1)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=1, shuffle=False, num_workers=1)

        # set up dataloaders dict
        dataloaders = {
            "train": data_loader_train,
            "val": data_loader_val
        }
      
        # Initialize the model
        model = Mars_Spectrometry_Model(1600, 1)
        model.to(device)
          
        # Define the loss function and optimizer
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.8, min_lr=1e-8)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

        model, best_acc, best_loss = train_model(device, model, criterion, optimizer, scheduler, dataloaders, num_epochs=25)

        # Process is complete.
        print(f"Training process for compound {compound} has finished")

        torch.save(model.state_dict(), f"checkpoints/compounds/model_BCEWithLogitsLoss_{compound}.ckpt")

        compounds_loss[compound] = best_loss

    print("Losses:")
    for compound in compounds:
        print(f"{compound} : {compounds_loss[compound]}")

    avg_loss = np.mean(list(compounds_loss.values()))
    print("\nAverage loss")
    print(avg_loss)


if __name__ == "__main__":
    model = train()

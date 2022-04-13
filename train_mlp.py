import numpy as np
import time
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import random

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset

from model import Mars_Spectrometry_Model, Soft_Ordering_1D_CNN, DNN
from dataset import Mars_Spectrometry_Dataset
from preprocessing import drop_frac_and_He, remove_background_abundance, \
    scale_abun, preprocess_sample, abun_per_tempbin
from tqdm import tqdm

from sklearn.model_selection import KFold


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def predict_for_sample(dataset_path, all_test_files, compounds_order, sample_id, model, device):
    # Import sample
    temp_sample = pd.read_csv(dataset_path + "/" + all_test_files[sample_id])

    # Preprocess sample
    temp_sample = preprocess_sample(temp_sample)

    # Feature engineering on sample
    temp_sample = abun_per_tempbin(temp_sample)

    # Generate predictions for each class
    temp_sample_preds_dict = {}

    preds = torch.sigmoid(model(torch.FloatTensor(temp_sample.values).to(device))).cpu().squeeze().tolist()

    for i, compound in enumerate(compounds_order):
        temp_sample_preds_dict[compound] = preds[i]

    return temp_sample_preds_dict


def generate_submission(model, device, prefix, seed, fold):
    model.eval()

    pd.set_option("max_colwidth", 80)

    dataset_path = "dataset"
    metadata = pd.read_csv(dataset_path + "/metadata.csv", index_col="sample_id")

    val_files = metadata[metadata["split"] == "val"]["features_path"].to_dict()
    test_files = metadata[metadata["split"] == "test"]["features_path"].to_dict()

    # Create dict with both validation and test sample IDs and paths
    all_test_files = val_files.copy()
    all_test_files.update(test_files)
    print("Total test files: ", len(all_test_files))

    # Import submission format
    submission_template_df = pd.read_csv(
        dataset_path + "/" + "submission_format.csv", index_col="sample_id"
    )

    compounds_order = submission_template_df.columns
    sample_order = submission_template_df.index

    # Dataframe to store submissions in
    final_submission_df = pd.DataFrame(
        [
            predict_for_sample(dataset_path, all_test_files, compounds_order, sample_id, model, device)
            for sample_id in tqdm(sample_order)
        ],
        index=sample_order,
    )

    print(final_submission_df.head())
    final_submission_df.to_csv(f"submission/{prefix}_{seed}_{fold}_submission.csv")


def train_model(device, model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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

                    loss = criterion(outputs, labels)

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

            if phase == "train":
                scheduler.step(epoch_loss)

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_acc
  
  
def train():
    # select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)

    kfold_n = 7
    seeds = [0, 1, 2, 3, 4, 5, 6]

    # initialize custom dataset and transformations
    dataset_train = Mars_Spectrometry_Dataset("dataset/train_features.pickle", "dataset/train_labels.pickle")
    dataset_val = Mars_Spectrometry_Dataset("dataset/val_features.pickle", "dataset/val_labels.pickle")
    dataset = ConcatDataset([dataset_train, dataset_val])

    num_features = dataset_train.num_features
    num_labels = dataset_train.num_labels

    """
    # number of validation samples (30% of training samples)
    num_val_samples = int(len(dataset_train) * 0.3)

    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-num_val_samples])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[-num_val_samples:])
    """

    total_acc = 0

    for seed in seeds:
        seed_everything(seed)

        # For fold results
        results = {}

        kfold = KFold(n_splits=7, shuffle=True, random_state=seed)

        # K-fold Cross Validation model evaluation
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):

            # Print
            print(f"Fold {fold} - seed {seed}")
            print("-" * 10)

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

            # define training and validation data loaders
            data_loader_train = torch.utils.data.DataLoader(
                dataset, batch_size=16, num_workers=1, sampler=train_subsampler)

            data_loader_val = torch.utils.data.DataLoader(
                dataset, batch_size=1, num_workers=1, sampler=val_subsampler)

            # set up dataloaders dict
            dataloaders = {
                "train": data_loader_train,
                "val": data_loader_val
            }
          
            # Initialize the model
            model = Mars_Spectrometry_Model(num_features, num_labels)
            model.to(device)
              
            # Define the loss function and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=4, factor=0.8, min_lr=1e-8)

            model, best_acc = train_model(device, model, criterion, optimizer, scheduler, dataloaders, num_epochs=24)

            print(f"Accuracy for fold {fold}: {best_acc}")
            print("-" * 10)

            results[fold] = best_acc

            #torch.save(model.state_dict(), f"checkpoints/model_BCEWithLogitsloss_fold{fold}_seed{seed}.ckpt")
            generate_submission(model, device, f"mlp_acc_{best_acc}", seed, fold)

        # Print fold results
        print(f"K-Fold Cross Validation results for {kfold_n} folds")
        print("-" * 10)

        avg = 0.0

        for key, value in results.items():
            print(f"Fold {key} seed {seed}: {value}")
            avg += value

        print(f"Average: {avg/len(results.items())}")

        total_acc += avg/len(results.items())

    print(f"Accuracy across all folds/seeds: {total_acc/len(seeds)}")


if __name__ == "__main__":
    model = train()

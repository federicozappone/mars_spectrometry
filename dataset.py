import numpy as np
import torch
import pandas as pd
import pickle

from torch.utils.data import Dataset, DataLoader


class Mars_Spectrometry_Dataset(Dataset):
    
    def __init__(self, train_features_path, train_labels_path):
        train_features_file = open(train_features_path, "rb")
        train_labels_file = open(train_labels_path, "rb")

        self.train_features = pickle.load(train_features_file)
        self.train_labels = pickle.load(train_labels_file)

        self.X_data = torch.FloatTensor(self.train_features.values)
        self.y_data = torch.FloatTensor(self.train_labels.values)
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


class Mars_Spectrometry_Dataset_Mono(Dataset):
    
    def __init__(self, train_features_path, train_labels_path, compound_index=0):
        self.compounds = "basalt", "carbonate", "chloride", "iron_oxide", "oxalate", "oxychlorine", "phyllosilicate", "silicate", "sulfate", "sulfide"
        self.selected = self.compounds[compound_index]

        train_features_file = open(train_features_path, "rb")
        train_labels_file = open(train_labels_path, "rb")

        self.train_features = pickle.load(train_features_file)
        self.train_labels = pickle.load(train_labels_file)[self.selected]

        self.X_data = torch.FloatTensor(self.train_features.values)
        self.y_data = torch.FloatTensor(self.train_labels.values)
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

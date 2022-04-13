import numpy as np
import torch
import pandas as pd
import pickle

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


class Mars_Spectrometry_Dataset(Dataset):
    
    def __init__(self, features_path, labels_path, use_pca=False):
        features_file = open(features_path, "rb")
        labels_file = open(labels_path, "rb")

        self.features = pickle.load(features_file)
        self.labels = pickle.load(labels_file)

        if use_pca is True:
            n_comp = 50

            pca_data = (PCA(n_components=n_comp, random_state=42).fit_transform(self.features))
            pca_data = pd.DataFrame(pca_data, columns=[f"pca_G-{i}" for i in range(n_comp)])

            print(pca_data.head())

            print(f"PCA: {pca_data.shape}")

            pca_data.reset_index(drop=True, inplace=True)
            self.features.reset_index(drop=True, inplace=True)

            self.features = pd.concat([self.features, pca_data], axis=1)

        self.X_data = torch.FloatTensor(self.features.values)
        self.y_data = torch.FloatTensor(self.labels.values)

        self.num_features = self.features.shape[1]
        self.num_labels = self.labels.shape[1]
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)


class Mars_Spectrometry_Dataset_Mono(Dataset):
    
    def __init__(self, train_features_path, train_labels_path, compound_index=0):
        self.compounds = "basalt", "carbonate", "chloride", "iron_oxide", "oxalate", "oxychlorine", "phyllosilicate", "silicate", "sulfate", "sulfide"
        self.selected = self.compounds[compound_index]

        features_file = open(features_path, "rb")
        labels_file = open(labels_path, "rb")

        self.features = pickle.load(features_file)
        self.labels = pickle.load(labels_file)[self.selected]

        self.X_data = torch.FloatTensor(self.features.values)
        self.y_data = torch.FloatTensor(self.labels.values)

        self.num_features = self.features.shape[1]
        self.num_labels = self.labels.shape[1]
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

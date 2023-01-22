import numpy as np
import torch
import pandas as pd
import pickle

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, \
    RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold



class Mars_Spectrometry_Dataset(Dataset):
    
    def __init__(self, features_path, labels_path, norm=None, use_pca=False, seed=0):
        features_file = open(features_path, "rb")
        labels_file = open(labels_path, "rb")

        self.features = pickle.load(features_file)
        self.labels = pickle.load(labels_file)

        if norm is not None:
            self.features, self.ss = self.norm_fit(self.features, norm)

        if use_pca is True:
            n_comp = 50

            pca_data = (PCA(n_components=n_comp, random_state=seed).fit_transform(self.features))
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

    def norm_fit(self, df, sc_name="zsco"):   
        ss_dic = {
                    "zsco": StandardScaler(),
                    "mima": MinMaxScaler(),
                    "maxb": MaxAbsScaler(), 
                    "robu": RobustScaler(),
                    "norm": Normalizer(), 
                    "quan": QuantileTransformer(n_quantiles=100, random_state=seed, output_distribution="normal"),
                    "powe": PowerTransformer()
                  }

        ss = ss_dic[sc_name]
        df_2 = pd.DataFrame(ss.fit_transform(df), index=df.index, columns=df.columns)
        
        return df_2, ss

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

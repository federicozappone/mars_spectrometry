import pandas as pd
import numpy as np
import pickle
import torch
import cv2
import albumentations as A

from pyDeepInsight import ImageTransformer, LogScaler
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader

from albumentations.pytorch import ToTensorV2


class Mars_Spectrometry_Image_Dataset(Dataset):
    
    def __init__(self, features_path, labels_path, image_size=128, seed=42):
        self.features = pickle.load(open(features_path, "rb"))
        self.labels = pickle.load(open(labels_path, "rb"))

        self.num_features = self.features.shape[1]
        self.num_labels = self.labels.shape[1]
        self.image_size = image_size

        self.labels = torch.Tensor(self.labels.values)

        tsne = TSNE(n_components=2, perplexity=2, metric="cosine",
                    random_state=seed, n_jobs=-1)

        it = ImageTransformer(feature_extractor=tsne, pixels=image_size)
        it.fit(self.features)

        self.images = it.transform(self.features)
        self.transforms = A.Compose([ToTensorV2()])
        
    def __getitem__(self, index):
        image = self.transforms(image=self.images[index])["image"]
        return image.to(torch.float), self.labels[index]
        
    def __len__ (self):
        return len(self.images)

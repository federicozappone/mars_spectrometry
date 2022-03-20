import itertools
import numpy as np
import pandas as pd
import time
import pickle
import os.path

from scipy.stats import uniform

from pandas_path import path
from tqdm import tqdm

from preprocessing import drop_frac_and_He, remove_background_abundance, scale_abun, preprocess_sample, abun_per_tempbin


pd.set_option("max_colwidth", 80)

dataset_path = "dataset"
metadata = pd.read_csv(dataset_path + "/metadata.csv", index_col="sample_id")

train_files = metadata[metadata["split"] == "train"]["features_path"].to_dict()
val_files = metadata[metadata["split"] == "val"]["features_path"].to_dict()
test_files = metadata[metadata["split"] == "test"]["features_path"].to_dict()

print("Number of training samples: ", len(train_files))
print("Number of validation samples: ", len(val_files))
print("Number of testing samples: ", len(test_files))

train_labels = pd.read_csv(dataset_path + "/train_labels.csv", index_col="sample_id")
val_labels = pd.read_csv(dataset_path + "/val_labels.csv", index_col="sample_id")

# Assembling preprocessed and transformed training set

train_features_dict = {}
print("Total number of train files: ", len(train_files))

for i, (sample_id, filepath) in enumerate(tqdm(train_files.items())):
    # Load training sample
    temp = pd.read_csv(dataset_path + "/" + filepath)

    # Preprocessing training sample
    train_sample_pp = preprocess_sample(temp)

    # Feature engineering
    train_sample_fe = abun_per_tempbin(train_sample_pp).reset_index(drop=True)
    train_features_dict[sample_id] = train_sample_fe

train_features = pd.concat(
    train_features_dict, names=["sample_id", "dummy_index"]
).reset_index(level="dummy_index", drop=True)


# Assembling preprocessed and transformed training set

val_features_dict = {}
print("Total number of val files: ", len(val_files))

for i, (sample_id, filepath) in enumerate(tqdm(val_files.items())):
    # Load training sample
    temp = pd.read_csv(dataset_path + "/" + filepath)

    # Preprocessing training sample
    val_sample_pp = preprocess_sample(temp)

    # Feature engineering
    val_sample_fe = abun_per_tempbin(val_sample_pp).reset_index(drop=True)
    val_features_dict[sample_id] = val_sample_fe

val_features = pd.concat(
    val_features_dict, names=["sample_id", "dummy_index"]
).reset_index(level="dummy_index", drop=True)


print("Train features")
print(train_features.head())

print("Val features")
print(val_features.head())

# Make sure that all sample IDs in features and labels are identical
assert train_features.index.equals(train_labels.index)
assert val_features.index.equals(val_labels.index)

pickle.dump(train_features, open("dataset/train_features.pickle", "wb"))
pickle.dump(train_labels, open("dataset/train_labels.pickle", "wb"))
pickle.dump(val_features, open("dataset/val_features.pickle", "wb"))
pickle.dump(val_labels, open("dataset/val_labels.pickle", "wb"))

import itertools
from pathlib import Path
from pprint import pprint
import time
import pickle
from scipy.stats import uniform
import torch

import os.path

from matplotlib import pyplot as plt, cm
import numpy as np
import pandas as pd
from pandas_path import path
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, \
    accuracy_score, make_scorer, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import ShuffleSplit

from xgboost import XGBClassifier


pd.set_option("max_colwidth", 80)
seed = 42  # For reproducibility

dataset_path = "dataset"
metadata = pd.read_csv(dataset_path + "/metadata.csv", index_col="sample_id")
print(metadata.head())

train_files = metadata[metadata["split"] == "train"]["features_path"].to_dict()
val_files = metadata[metadata["split"] == "val"]["features_path"].to_dict()
test_files = metadata[metadata["split"] == "test"]["features_path"].to_dict()

print("Number of training samples: ", len(train_files))
print("Number of validation samples: ", len(val_files))
print("Number of testing samples: ", len(test_files))

train_labels = pd.read_csv(dataset_path + "/train_labels.csv", index_col="sample_id")
print(train_labels.head())


# Select sample IDs for five commercial samples and five testbed samples
sample_id_commercial = (
    metadata[metadata["instrument_type"] == "commercial"]
    .index
    .values[0:5]
)
sample_id_testbed = (
    metadata[metadata["instrument_type"] == "sam_testbed"]
    .index
    .values[0:5]
)


# Import sample files for EDA
sample_commercial_dict = {}
sample_testbed_dict = {}

for i in range(0, 5):
    comm_lab = sample_id_commercial[i]
    sample_commercial_dict[comm_lab] = pd.read_csv(dataset_path + "/" + train_files[comm_lab])

    test_lab = sample_id_testbed[i]
    sample_testbed_dict[test_lab] = pd.read_csv(dataset_path + "/" + train_files[test_lab])


"""
# For commercial
fig, ax = plt.subplots(1, 5, figsize=(15, 3), constrained_layout=True)
fig.suptitle("Commercial samples")
fig.supxlabel("Temperature")
fig.supylabel("Abundance")

for i in range(0, 5):
    sample_lab = sample_id_commercial[i]
    sample_df = sample_commercial_dict[sample_lab]

    plt.subplot(1, 5, i + 1)
    for m in sample_df["m/z"].unique():
        plt.plot(
            sample_df[sample_df["m/z"] == m]["temp"],
            sample_df[sample_df["m/z"] == m]["abundance"],
        )

    plt.title(sample_lab)

# For testbed
fig, ax = plt.subplots(1, 5, figsize=(15, 3), constrained_layout=True)
fig.suptitle("SAM testbed samples")
fig.supxlabel("Temperature")
fig.supylabel("Abundance")

for i in range(0, 5):
    sample_lab = sample_id_testbed[i]
    sample_df = sample_testbed_dict[sample_lab]

    plt.subplot(1, 5, i + 1)
    for m in sample_df["m/z"].unique():
        plt.plot(
            sample_df[sample_df["m/z"] == m]["temp"],
            sample_df[sample_df["m/z"] == m]["abundance"],
        )
    plt.title(sample_lab)

plt.show()
"""



# preprocessing

# Selecting a testbed sample to demonstrate preprocessing steps
sample_lab = sample_id_testbed[1]
sample_df = sample_testbed_dict[sample_lab]


def drop_frac_and_He(df):
    """
    Drops fractional m/z values, m/z values > 100, and carrier gas m/z

    Args:
        df: a dataframe representing a single sample, containing m/z values

    Returns:
        The dataframe without fractional an carrier gas m/z
    """

    # drop fractional m/z values
    df = df[df["m/z"].transform(round) == df["m/z"]]
    assert df["m/z"].apply(float.is_integer).all(), "not all m/z are integers"

    # drop m/z values greater than 99
    df = df[df["m/z"] < 100]

    # drop carrier gas
    df = df[df["m/z"] != 4]

    return df


def remove_background_abundance(df):
    """
    Subtracts minimum abundance value

    Args:
        df: dataframe with 'm/z' and 'abundance' columns

    Returns:
        dataframe with minimum abundance subtracted for all observations
    """

    df["abundance_minsub"] = df.groupby(["m/z"])["abundance"].transform(
        lambda x: (x - x.min())
    )

    return df

def scale_abun(df):
    """
    Scale abundance from 0-100 according to the min and max values across entire sample

    Args:
        df: dataframe containing abundances and m/z

    Returns:
        dataframe with additional column of scaled abundances
    """

    df["abun_minsub_scaled"] = minmax_scale(df["abundance_minsub"].astype(float))

    return df

# Preprocess function
def preprocess_sample(df):
    df = drop_frac_and_He(df)
    df = remove_background_abundance(df)
    df = scale_abun(df)
    return df


# feature engineering

# Create a series of temperature bins
temprange = pd.interval_range(start=-100, end=1500, freq=100)
temprange

# Make dataframe with rows that are combinations of all temperature bins
# and all m/z values
allcombs = list(itertools.product(temprange, [*range(0, 100)]))

allcombs_df = pd.DataFrame(allcombs, columns=["temp_bin", "m/z"])
allcombs_df.head()


def abun_per_tempbin(df):

    """
    Transforms dataset to take the preprocessed max abundance for each
    temperature range for each m/z value

    Args:
        df: dataframe to transform

    Returns:
        transformed dataframe
    """

    # Bin temperatures
    df["temp_bin"] = pd.cut(df["temp"], bins=temprange)

    # Combine with a list of all temp bin-m/z value combinations
    df = pd.merge(allcombs_df, df, on=["temp_bin", "m/z"], how="left")

    # Aggregate to temperature bin level to find max
    df = df.groupby(["temp_bin", "m/z"]).max("abun_minsub_scaled").reset_index()

    # Fill in 0 for abundance values without information
    df = df.replace(np.nan, 0)

    # Reshape so each row is a single sample
    df = df.pivot_table(columns=["m/z", "temp_bin"], values=["abun_minsub_scaled"])

    return df

"""
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


print("Train features")
print(train_features.head())

# Make sure that all sample IDs in features and labels are identical
assert train_features.index.equals(train_labels.index)

with open("dataset/train_features.pickle", "wb") as pickle_file:
    pickle.dump(train_features, pickle_file)
"""

# open a file, where you stored the pickled data
train_features_file = open("dataset/train_features.pickle", "rb")

# dump information to that file
train_features = pickle.load(train_features_file)

# open a file, where you stored the pickled data
train_labels_file = open("dataset/train_labels.pickle", "rb")

# dump information to that file
train_labels = pickle.load(train_labels_file)

print("Train features")
print(train_features.head())

# training

# Define stratified k-fold validation
skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)

# Define log loss
log_loss_scorer = make_scorer(log_loss, needs_proba=True)


# Check log loss score for baseline dummy model
def logloss_cross_val(clf, X, y):

    # Generate a score for each label class
    log_loss_cv = {}
    for col in y.columns:
        print("Processing col", col)

        y_col = y[col]  # take one label at a time
        log_loss_cv[col] = np.mean(
            cross_val_score(clf, X.values, y_col, cv=skf, scoring=log_loss_scorer)
        )

    avg_log_loss = np.mean(list(log_loss_cv.values()))

    return log_loss_cv, avg_log_loss


# Dummy classifier
dummy_clf = DummyClassifier(strategy="prior")

print("Dummy model log-loss:")
dummy_logloss = logloss_cross_val(dummy_clf, train_features, train_labels)
pprint(dummy_logloss[0])
print("\nAverage log-loss")
print(dummy_logloss[1])

print()
print("Train features:\n", train_features.to_numpy()[0])
print("Train features shape:\n", len(train_features.to_numpy()[0]))
print("Train labels:\n", train_labels.to_numpy())
print("Train labels shape:\n", len(train_labels.to_numpy()[0]))
print()

# Define logistic regression model
logreg_clf = LogisticRegression(penalty="l1", solver="liblinear", C=10, random_state=seed)

print("Logistic regression model log-loss:\n")
logreg_logloss = logloss_cross_val(logreg_clf, train_features, train_labels)
pprint(logreg_logloss[0])
print("Average log-loss")
print(logreg_logloss[1])

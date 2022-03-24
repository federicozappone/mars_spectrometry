import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn import model_selection, preprocessing, metrics
from sklearn.metrics import make_scorer, log_loss

from tqdm import tqdm
from pprint import pprint



train_features = pickle.load(open("dataset/train_features.pickle", "rb"))
train_labels = pickle.load(open("dataset/train_labels.pickle", "rb"))

val_features = pickle.load(open("dataset/val_features.pickle", "rb"))
val_labels = pickle.load(open("dataset/val_labels.pickle", "rb"))

# Define stratified k-fold validation
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

# Define log loss
log_loss_scorer = make_scorer(log_loss, needs_proba=True)

# Check log loss score for baseline dummy model
def logloss_cross_val(clf, X, y):
    # Generate a score for each label class
    log_loss_cv = {}
    for col in y.columns:

        y_col = y[col]  # take one label at a time
        log_loss_cv[col] = np.mean(
            cross_val_score(clf, X.values, y_col, cv=skf, scoring=log_loss_scorer)
        )

    avg_log_loss = np.mean(list(log_loss_cv.values()))

    return log_loss_cv, avg_log_loss


# grid of parameters
grid_params = {
    'learning_rate': [0.001],
    'num_leaves': [50, 100, 200],
    'max_depth' : [5, 6, 7, 8],
    'random_state' : [42],
    'subsample' : [0.5, 0.7, 0.8],
    'metric':['logloss']
}


# modelling
clf = lgb.LGBMClassifier()
grid = RandomizedSearchCV(clf,grid_params, verbose=1, cv=3, n_jobs=-1, n_iter=10)

print("Lightgbm model log-loss:\n")
logreg_logloss = logloss_cross_val(grid, train_features, train_labels)

pprint(logreg_logloss[0])
print("Average log-loss")
print(logreg_logloss[1])

# Train logistic regression model with l1 regularization, where C = 10

# Initialize dict to hold fitted models
fitted_logreg_dict = {}

# Split into binary classifier for each class
for col in train_labels.columns:

    y_train_col = train_labels[col]  # Train on one class at a time

    # output the trained model, bind this to a var, then use as input
    # to prediction function
    clf = lgb.LGBMClassifier(random_state=42)
    fitted_logreg_dict[col] = clf.fit(train_features.values, y_train_col) # Train

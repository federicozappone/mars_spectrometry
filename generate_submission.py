import numpy as np
import time
import os
import copy
import torch
import pandas as pd

from model import Mars_Spectrometry_Model
from preprocessing import drop_frac_and_He, remove_background_abundance, scale_abun, preprocess_sample, abun_per_tempbin
from tqdm import tqdm


def predict_for_sample(dataset_path, all_test_files, compounds_order, sample_id, model, device):
    model.eval()

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


def generate_submission(model_state_dict_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = Mars_Spectrometry_Model(1600, 10)
    model.load_state_dict(torch.load(model_state_dict_path))
    model.to(device)

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
    final_submission_df.to_csv("submission/submission.csv")


if __name__ == "__main__":
    generate_submission("checkpoints/model_BCEWithLogitsloss.ckpt")
import numpy as np
import time
import os
import copy
import torch
import pandas as pd

from model import Mars_Spectrometry_Model
from preprocessing import drop_frac_and_He, remove_background_abundance, scale_abun, preprocess_sample, abun_per_tempbin
from tqdm import tqdm


def predict_for_sample(dataset_path, all_test_files, compounds_order, sample_id, models, device):
    # Import sample
    temp_sample = pd.read_csv(dataset_path + "/" + all_test_files[sample_id])

    # Preprocess sample
    temp_sample = preprocess_sample(temp_sample)

    # Feature engineering on sample
    temp_sample = abun_per_tempbin(temp_sample)

    # Generate predictions for each class
    temp_sample_preds_dict = {}

    for compound in compounds_order:
        pred = torch.sigmoid(models[compound](torch.FloatTensor(temp_sample.values).to(device))).cpu().squeeze().tolist()
        temp_sample_preds_dict[compound] = pred

    return temp_sample_preds_dict


def generate_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Initialize the models
    models = {}

    for compound in compounds_order:
        models[compound] = Mars_Spectrometry_Model(1600, 1)
        models[compound].load_state_dict(torch.load(f"checkpoints/compounds/model_BCEWithLogitsLoss_{compound}.ckpt"))
        models[compound].to(device)
        models[compound].eval()

    # Dataframe to store submissions in
    final_submission_df = pd.DataFrame(
        [
            predict_for_sample(dataset_path, all_test_files, compounds_order, sample_id, models, device)
            for sample_id in tqdm(sample_order)
        ],
        index=sample_order,
    )

    print(final_submission_df.head())
    final_submission_df.to_csv("submission/submission_compounds.csv")


if __name__ == "__main__":
    generate_submission()

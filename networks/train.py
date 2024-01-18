# -*- coding: utf-8 -*-

import os, gc, shutil

import pandas as pd
import numpy as np
from tensorflow import keras

import config
from data_processing import (
    get_folds_filenames,
    get_datafolds_filenames,
    get_fold_vocabularies,
    save_w2i_dictionary,
    load_dictionaries,
    train_data_generator,
)
from networks.models import build_models
from networks.test import evaluate_model
from kaldi_preprocessing import *


# Utility function for training, validating, and testing a model and saving the logs in a CSV file
def train_and_test_model(
    data, vocabularies, epochs, model, prediction_model, pred_model_filepath, log_path
):
    train_images, train_labels, val_images, val_labels, test_images, test_labels = data
    w2i, i2w = vocabularies

    # Instantiate logs variables
    loss_acc = []
    val_symer_acc = []
    val_seqer_acc = []

    # Train and validate
    best_symer = np.Inf
    best_epoch = 0
    for epoch in range(epochs):
        print(f"--Epoch {epoch + 1}--")
        print("Training:")
        history = model.fit(
            train_data_generator(train_images, train_labels, w2i),
            epochs=1,
            verbose=2,
            steps_per_epoch=len(train_images) // config.batch_size,
        )
        loss_acc.extend(history.history["loss"])
        print("Validating:")
        val_symer, val_seqer = evaluate_model(
            prediction_model, val_images, val_labels, i2w
        )[0:2]
        val_symer_acc.append(val_symer)
        val_seqer_acc.append(val_seqer)
        if val_symer < best_symer:
            best_symer = val_symer
            best_epoch = epoch
            print(f"Saving new best prediction model to file {pred_model_filepath}")
            prediction_model.save(filepath=pred_model_filepath)
    print(f"Best validation SymER (%): {best_symer:.2f} at epoch {best_epoch + 1}")

    # Test the best validation model
    print("Evaluating best validation model over test data")
    prediction_model = keras.models.load_model(pred_model_filepath)
    test_symer, test_seqer, test_data = evaluate_model(
        prediction_model, test_images, test_labels, i2w
    )

    # Save fold logs
    # The last line on the CSV file is the one corresponding to the best validation model
    loss_acc.extend(["-", loss_acc[best_epoch]])
    val_symer_acc.extend(["-", val_symer_acc[best_epoch]])
    val_seqer_acc.extend(["-", val_seqer_acc[best_epoch]])
    logs = {
        "loss": loss_acc,
        "val_symer": val_symer_acc,
        "val_seqer": val_seqer_acc,
        "test_symer": ["-"] * (len(val_symer_acc) - 1) + [test_symer],
        "test_seqer": ["-"] * (len(val_seqer_acc) - 1) + [test_seqer],
    }
    logs = pd.DataFrame.from_dict(logs)
    logs.to_csv(log_path, index=False)

    return test_data

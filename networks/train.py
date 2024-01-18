from typing import List, Dict, Tuple

import pandas as pd
from tensorflow import keras

from networks.test import evaluate_model
from my_utils.data_processing import train_data_generator


# Utility function for training, validating, and testing a model and saving the logs in a CSV file
def train_and_test_model(
    task: str,
    data: Tuple[List[str], List[str], List[str], List[str], List[str], List[str]],
    vocabularies: Tuple[Dict[str, int], Dict[int, str]],
    epochs: int,
    batch_size: int,
    model: keras.Model,
    prediction_model: keras.Model,
    pred_model_filepath: str,
    log_path: str,
):
    train_images, train_labels, val_images, val_labels, test_images, test_labels = data
    w2i, i2w = vocabularies

    # Instantiate logs variables
    loss_acc = []
    val_symer_acc = []
    val_seqer_acc = []

    # Train and validate
    best_symer = float("inf")
    best_epoch = 0
    for epoch in range(epochs):
        print(f"--Epoch {epoch + 1}--")
        print("Training:")
        history = model.fit(
            train_data_generator(
                task=task,
                images_files=train_images,
                labels_files=train_labels,
                w2i=w2i,
                batch_size=batch_size,
            ),
            epochs=1,
            verbose=2,
            steps_per_epoch=len(train_images) // batch_size,
        )
        loss_acc.extend(history.history["loss"])
        print("Validating:")
        val_symer, val_seqer = evaluate_model(
            task=task,
            model=prediction_model,
            images_files=val_images,
            labels_files=val_labels,
            i2w=i2w,
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
        task=task,
        model=prediction_model,
        images_files=test_images,
        labels_files=test_labels,
        i2w=i2w,
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

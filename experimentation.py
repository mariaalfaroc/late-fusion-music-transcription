import os
import gc

import pandas as pd
from tensorflow import keras

from my_utils.kaldi_preprocessing import *
from networks.models import build_models
from networks.train import train_and_test_model
from networks.test import evaluate_model
from scenarios.folds_creation import (
    get_folds_filenames,
    get_datafold_filenames,
    check_and_retrive_vocabulary,
)


# Utility function for performing a k-fold cross-validation
# experiment on a single dataset (SCENARIOS A, B, and D)
def k_fold_experiment(
    *,
    task: str,
    scenario_name: str,
    epochs: int = 150,
    batch_size: int = 16,
):
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print(f"5-fold cross-validation experiment for scenario {scenario_name}")
    print(f"\tTask: {task}")
    print(f"\tEpochs: {epochs}")
    print(f"\tBatch size: {batch_size}")

    # ---------- FOLDS COLLECTION

    folds = get_folds_filenames(scenario_name)

    # ---------- 5-FOLD EVALUATION

    for id, (train_fold, val_fold, test_fold) in enumerate(
        zip(folds["train"], folds["val"], folds["test"])
    ):
        # With 'clear_session()' called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time
        keras.backend.clear_session()
        gc.collect()

        print(f"Fold {id}")

        # Get the current fold data
        train_images, train_labels = get_datafold_filenames(
            task=task, fold_filename=train_fold
        )
        val_images, val_labels = get_datafold_filenames(
            task=task, fold_filename=val_fold
        )
        test_images, test_labels = get_datafold_filenames(
            task=task, fold_filename=test_fold
        )
        print(f"Train size: {len(train_images)}")
        print(f"Validation size: {len(val_images)}")
        print(f"Test size: {len(test_images)}")

        # Check and retrieve vocabulary
        w2i, i2w, w2i_path = check_and_retrive_vocabulary(
            fold_id=id, retrieve_w2i_path=True
        )

        # Build the models
        model, prediction_model = build_models(task=task, num_labels=len(w2i))

        # Set filepaths outputs
        output_dir = f"results/scenario{scenario_name}/fold{id}"
        os.makedirs(output_dir, exist_ok=True)
        pred_model_filepath = os.path.join(output_dir, f"best_{task}_model.keras")
        log_path = os.path.join(output_dir, f"{task}_logs.csv")

        # Train, validate, and test models
        # Save logs in CSV file
        test_data = train_and_test_model(
            task=task,
            data=(
                train_images,
                train_labels,
                val_images,
                val_labels,
                test_images,
                test_labels,
            ),
            vocabularies=(w2i, i2w),
            epochs=epochs,
            batch_size=batch_size,
            model=model,
            prediction_model=prediction_model,
            pred_model_filepath=pred_model_filepath,
            log_path=log_path,
        )

        # Save to Kaldi format
        # test_data = [y_true_acc, raw_y_pred_acc, y_pred_len_acc]
        kaldi_dir = os.path.join(output_dir, "kaldi")
        os.makedirs(kaldi_dir, exist_ok=True)
        write_kaldi_fold(fold_filepath=test_fold, kaldi_dir=kaldi_dir, fold_type="test")
        write_kaldi_vocabulary(w2i_filepath=w2i_path, kaldi_dir=kaldi_dir)
        write_kaldi_groundtruth(
            kaldi_dir=kaldi_dir, id_labels=test_labels, labels=test_data[0]
        )
        write_kaldi_confmat(
            kaldi_dir=kaldi_dir,
            fold_type=f"test-{task}",
            id_preds=test_labels,
            preds=test_data[1],
            preds_len=test_data[2],
        )

        # Clear memory
        del train_images, train_labels, val_images, val_labels, test_images, test_labels
        del test_data
        del model, prediction_model


# Utility function for performing a k-fold test partition experiment using previously trained models
def k_fold_test_experiment():
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print("k-fold test performance experiment")
    print(f"Data used {config.base_dir.stem}")

    # ---------- DATA COLLECTION

    test_folds_files = get_folds_filenames("test")
    test_images_fnames, test_labels_fnames = get_datafolds_filenames(test_folds_files)

    # ---------- K-FOLD EVALUATION

    # Start the k-fold evaluation scheme
    k = len(test_images_fnames)
    for i in range(k):
        # With 'clear_session()' called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time.
        keras.backend.clear_session()
        gc.collect()

        print(f"Fold {i}")

        # Set filepaths outputs
        output_dir = config.output_dir / config.task / f"Fold{i}"
        pred_model_filepath = output_dir / "best_model.keras"
        w2i_filepath = output_dir / "w2i.json"
        log_path = output_dir / "test_logs.csv"

        # Get the current fold data
        test_images, test_labels = test_images_fnames[i], test_labels_fnames[i]
        assert len(test_images) == len(test_labels)
        print(f"Test: {len(test_images)}")

        # Load dictionary
        i2w = load_dictionaries(w2i_filepath)[1]

        # Test the best validation model
        print("Evaluating best validation model over test data")
        prediction_model = keras.models.load_model(pred_model_filepath)
        test_symer, test_seqer, test_data = evaluate_model(
            prediction_model, test_images, test_labels, i2w
        )

        # Save fold logs
        logs = {"test_symer": [test_symer], "test_seqer": [test_seqer]}
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(log_path, index=False)

        # Kaldi preprocessing
        kaldi_dir = output_dir / "kaldi"
        os.makedirs(kaldi_dir, exist_ok=True)
        # test_data = [y_true_acc, raw_y_pred_acc, y_pred_len_acc]
        kaldi_fold(
            fold_filepath=test_folds_files[i], kaldi_dir=kaldi_dir, fold_type="test"
        )
        kaldi_vocabulary(w2i_filepath=w2i_filepath, kaldi_dir=kaldi_dir)
        kaldi_groundtruth(
            kaldi_dir=kaldi_dir, id_labels=test_labels, labels=test_data[0]
        )
        kaldi_confmat(
            kaldi_dir,
            fold_type="test",
            id_preds=test_labels,
            preds=test_data[1],
            preds_len=test_data[2],
        )

        # Clear memory
        del test_images, test_labels
        del prediction_model
        del test_data

    return

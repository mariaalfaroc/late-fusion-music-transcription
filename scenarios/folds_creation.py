# -*- coding: utf-8 -*-

import os, gc, random, shutil, pathlib

import numpy as np
from tensorflow import keras

import config
from data_processing import get_folds_filenames, get_datafolds_filenames, load_dictionaries, preprocess_image, preprocess_label
from evaluation import ctc_greedy_decoder, compute_metrics

# We consider three levels of model performance: High (SER ~ 25-28 %), Medium (SER ~ 15-18 %), Low (SER ~ 5-8 %)
# We then evaluate all possible combinations of those three levels with OMR and AMT models

# There are nine scenarios:
# 1) OMR High - AMT High
# 2) OMR High - AMT Medium
# 3) OMR High - AMT Low
# 4) OMR Medium - AMT High
# 5) OMR Medium - AMT Medium
# 6) OMR Medium - AMT Low
# 7) OMR Low - AMT High
# 8) OMR Low - AMT Medium
# 9) OMR Low - AMT Low

#                   Scenario 1                              Scenario 2                              Scenario 3
#               OMR	                AMT                 OMR	                AMT                 OMR	                AMT              
# Train	    2.5% Part. Orig.    Part. Orig.         2.2% Part. Orig.      Part. Orig.       2.0% Part. Orig.    Part. Orig.
# Val	    Part. Orig.	        Part. Orig.         Part. Orig.         Part. Orig.         Part. Orig.         Part. Orig.
# Test	    Part. Orig.	        Part. Orig.         New set 1	        New set 1           New set 2	        New set 2 

#                   Scenario 4                              Scenario 5                              Scenario 6
#               OMR	                AMT                 OMR	                AMT                 OMR	                AMT              
# Train	    4.0% Part. Orig.    Part. Orig.         3.7% Part. Orig.      Part. Orig.       3.0% Part. Orig.    Part. Orig.
# Val	    Part. Orig.	        Part. Orig.         Part. Orig.         Part. Orig.         Part. Orig.         Part. Orig.
# Test	    Part. Orig.	        Part. Orig.         New set 1	        New set 1           New set 2	        New set 2 

#                   Scenario 7                              Scenario 8                              Scenario 9
#               OMR	                AMT                 OMR	                AMT                 OMR	                AMT              
# Train	    10.5% Part. Orig.   Part. Orig.         9.0% Part. Orig.    Part. Orig.         6.3% Part. Orig.    Part. Orig.
# Val	    Part. Orig.	        Part. Orig.         Part. Orig.         Part. Orig.         Part. Orig.         Part. Orig.
# Test	    Part. Orig.	        Part. Orig.         New set 1	        New set 1           New set 2	        New set 2 

#                   Scenario X
#               OMR	           AMT            
# Train	    Part. Orig.    Part. Orig.
# Val	    Part. Orig.	   Part. Orig.
# Test	    Part. Orig.	   Part. Orig.

# New set 1 == Created according to AMT model performace -> Samples of the original test partition whose Symbol Error Rate is lower than or equal to 30% 
# New set 2 == Created according to AMT model performace -> Samples of the original test partition whose Symbol Error Rate is lower than or equal to 10% 

# -------------------- SCENARIOS 1, 4, AND 7

# Utility function for creating 5-folds with train, validation, and test partitions using a subset of the train partition
def create_folds(p_size: float, scenario: str):
    # Obtain folds for ScenarioX
    config.set_scenario(value="X")
    print(f"Scenario{scenario} uses train ({p_size} %, only for OMR), val, and test partitions of Scenario{config.scenario}") 
    train_folds_files = get_folds_filenames("train")
    val_folds_files = get_folds_filenames("val")
    test_folds_files = get_folds_filenames("test")
    # Create Scenario{scenario} folder
    os.makedirs(str(config.folds_dir).replace("ScenarioX", f"Scenario{scenario}"), exist_ok=True)
    # Copy val and test folds
    for val, test in zip(val_folds_files, test_folds_files):
        shutil.copyfile(val, val.replace("ScenarioX", f"Scenario{scenario}"))
        shutil.copyfile(test, test.replace("ScenarioX", f"Scenario{scenario}"))
    # Create new train folds
    for i in train_folds_files:
        data = open(i).readlines()
        random.shuffle(data)
        new_size = int(len(data) * p_size / 100)
        data = data[:new_size]
        data = data[:-1] + [data[-1].split("\n")[0]]
        with open(i.replace("ScenarioX", f"Scenario{scenario}"), "w") as txt:
            for s in data:
                txt.write(s)
    return

# -------------------- SCENARIOS 2, 3, 5, 6, 8, AND 9

# Utility function for writing 5-folds with train, validation, and test partitions based on both model performance and subset of training samples
def write_folds(samples: list, p_size: float, scenario: str):
    # Obtain folds for ScenarioX
    config.set_scenario(value="X")
    train_folds_files = get_folds_filenames("train")
    val_folds_files = get_folds_filenames("val")
    # Create Scenario{scenario} folder
    os.makedirs(str(config.folds_dir).replace("ScenarioX", f"Scenario{scenario}"), exist_ok=True)
    # Create new train folds
    for i in train_folds_files:
        data = open(i).readlines()
        random.shuffle(data)
        new_size = int(len(data) * p_size / 100)
        data = data[:new_size]
        data = data[:-1] + [data[-1].split("\n")[0]]
        with open(i.replace("ScenarioX", f"Scenario{scenario}"), "w") as txt:
            for s in data:
                txt.write(s)
    # Copy val folds
    for val in val_folds_files:
        shutil.copyfile(val, val.replace("ScenarioX", f"Scenario{scenario}"))
    # Create new test folds
    config.set_scenario(value=scenario)
    for id, test in enumerate(samples):
        test_fold = os.path.join(config.folds_dir, f"test_gt_fold{id}.dat")
        # Write folds files
        with open(test_fold, "w") as txt:
            test = [s + "\n" for s in test[:-1]] + [test[-1]]
            txt.writelines(test)
    return

# Utility function for evaluating a model over a dataset and adding the samples that are lower or equal than a threshold to a list
def evaluate_model(model, images_files, labels_files, i2w, symer_threshold=30):
    new_set = []
    # Iterate over images
    for i in range(len(images_files)):
        images, images_len = list(zip(*[preprocess_image(images_files[i])]))
        images = np.array(images, dtype="float32")
        # Obtain predictions
        y_pred = model(images, training=False)
        # CTC greedy decoder (merge repeated, remove blanks, and i2w conversion)
        y_pred = ctc_greedy_decoder(y_pred, images_len, i2w)
        # Obtain true labels
        y_true = [preprocess_label(labels_files[i], training=False, w2i=None)]
        # Compute Symbol Error Rate
        symer = compute_metrics(y_true, y_pred)[0]
        # If the Symbol Error Rate is lower than or equal to the threshold, the sample gets added to the new subset
        if symer <= symer_threshold:
            new_set.append(pathlib.Path(labels_files[i]).stem)
    print(f"For this fold, only {len(new_set)} samples have a Symbol Error Rate lower than or equal to {symer_threshold}")
    return new_set

# Utility function for obtaining model predictions and creating a new subset based on the corresponding error prediction
def create_folds_according_ser(p_size: float, scenario: str, symer_threshold=30):
    keras.backend.clear_session()
    gc.collect()

    config.set_scenario(value="X")
    config.set_task(value="amt")
    config.set_data_globals()
    config.set_arch_globals(batch=4)

    # ---------- PRINT EXPERIMENT DETAILS

    print(f"Creating folds according on AMT model performance on Scenario{config.scenario}: Sym-Er Threshold = {symer_threshold}")
    print(f"Percentage of the new train partition (for OMR) = {p_size}")
    print(f"Data used {config.base_dir.stem}")

    # ---------- DATA COLLECTION

    test_folds_files = get_folds_filenames("test")
    test_images_fnames, test_labels_fnames = get_datafolds_filenames(test_folds_files) 

    # ---------- K-FOLD EVALUATION
    new_set = []

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

        # Get the current fold data
        test_images, test_labels = test_images_fnames[i], test_labels_fnames[i]
        assert len(test_images) == len(test_labels)
        print(f"Test: {len(test_images)}")

        # Get and save vocabularies
        i2w = load_dictionaries(w2i_filepath)[1]

        # Get prediction model
        prediction_model = keras.models.load_model(pred_model_filepath)

        # Evaluate model and add to the new set the samples whose prediction error is lower than or equal to threshold
        new_set.append(evaluate_model(prediction_model, test_images, test_labels, i2w, symer_threshold=symer_threshold))

        # Clear memory
        del test_images, test_labels
        del prediction_model

    # Create 5-folds using new set samples
    write_folds(samples=new_set, p_size=p_size, scenario=scenario)

    return

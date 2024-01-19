import os
import gc
import json
import shutil
import random
from typing import List, Dict, Tuple

from tensorflow import keras

from networks.test import evaluate_model


INPUT_EXTENSION = {"omr": "_distorted.jpg", "amt": ".wav"}
LABEL_EXTENSION = ".semantic"
VOCABS_DIR = "scenarios/vocabs/"
os.makedirs(VOCABS_DIR, exist_ok=True)


######################################################################################################################################################################################

# We consider three levels of model performance:
# 1) High (SER ~ 25-28 %)
# 2) Medium (SER ~ 15-18 %)
# 3) Low (SER ~ 5-8 %)
# We then evaluate all possible combinations of those three levels with OMR and AMT models

# Therefore, there are nine scenarios:
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
# Train	    2.5% Part. Orig.    Part. Orig.         2.2% Part. Orig.    Part. Orig.         2.0% Part. Orig.    Part. Orig.
# Val	    Part. Orig.	        Part. Orig.         Part. Orig.         Part. Orig.         Part. Orig.         Part. Orig.
# Test	    Part. Orig.	        Part. Orig.         New set 1	        New set 1           New set 2	        New set 2

#                   Scenario 4                              Scenario 5                              Scenario 6
#               OMR	                AMT                 OMR	                AMT                 OMR	                AMT
# Train	    4.0% Part. Orig.    Part. Orig.         3.7% Part. Orig.    Part. Orig.         3.0% Part. Orig.    Part. Orig.
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

######################################################################################################################################################################################


# Get all the folds filenames for each data partition
# folds = {"train": [".../train_gt_fold0.dat", ".../train_gt_fold1.dat", ...], "val": [...], "test": [...]}
def get_folds_filenames(scenario_name: str) -> Dict[str, List[str]]:
    scenario_dir = f"scenarios/Scenario{scenario_name}"

    folds = {"train": [], "val": [], "test": []}
    for fname in os.listdir(scenario_dir):
        if fname.startswith("train"):
            folds["train"].append(os.path.join(scenario_dir, fname))
        elif fname.startswith("val"):
            folds["val"].append(os.path.join(scenario_dir, fname))
        elif fname.startswith("test"):
            folds["test"].append(os.path.join(scenario_dir, fname))

    assert (
        len(folds["train"]) == len(folds["val"]) == len(folds["test"])
    ), "Folds are not balanced!"

    return {k: sorted(v) for k, v in folds.items()}


# Get all images and labels filenames
# of a corresponding fold filename
def get_datafold_filenames(
    task: str, fold_filename: list
) -> Tuple[List[str], List[str]]:
    images_filenames = []
    labels_filenames = []
    with open(fold_filename) as f:
        lines = f.read().splitlines()
    for line in lines:
        common_path = f"dataset/Corpus/{line}/{line}"
        images_filenames.append(common_path + INPUT_EXTENSION[task])
        labels_filenames.append(common_path + LABEL_EXTENSION)
    return images_filenames, labels_filenames


def check_and_retrive_vocabulary(fold_id: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    w2i_path = os.path.join(VOCABS_DIR, f"w2i_fold{fold_id}.json")
    if os.path.exists(w2i_path):
        w2i, i2w = load_dictionaries(filepath=w2i_path)
    else:
        # Use ScenarioD train files
        folds = get_folds_filenames("D")
        _, labels_filenames = get_datafold_filenames(
            task="omr", fold_filename=folds["train"][fold_id]
        )
        w2i, i2w = get_fold_vocabularies(labels_filenames)
        save_w2i_dictionary(w2i, filepath=w2i_path)
    return w2i, i2w


# Get dictionaries for w2i and i2w conversion
# corresponding to a single training fold
def get_fold_vocabularies(
    train_labels_fnames: List[str],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    # Get all tokens related to a SINGLE train data fold
    tokens = []
    for fname in train_labels_fnames:
        with open(fname) as f:
            tokens.extend(f.read().split())
    # Eliminate duplicates and sort them alphabetically
    tokens = sorted(set(tokens))
    # Create vocabularies
    w2i = dict(zip(tokens, range(len(tokens))))
    i2w = dict(zip(range(len(tokens)), tokens))
    return w2i, i2w


# Utility function for saving w2i dictionary in a JSON file
def save_w2i_dictionary(w2i: Dict[str, int], filepath: str):
    # Save w2i dictionary to JSON filepath to later retrieve it
    # No need to save both of them as they are related
    with open(filepath, "w") as json_file:
        json.dump(w2i, json_file)


# Retrieve w2i and i2w dictionaries from w2i JSON file
def load_dictionaries(filepath: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(filepath, "r") as json_file:
        w2i = json.load(json_file)
    i2w = {int(v): k for k, v in w2i.items()}
    return w2i, i2w


##################################### SCENARIOS 1, 4 AND 7:

# OMR data:
# For each train partition of Scenario X,
# we randomly select either 2.5% or 4% or a 10.5%, as corresponds
# Those are the new train partitions
# Validation and test partitions are those of Scenario X


# Utility function for checking if a scenario already exists
def check_scenario_exists(scenario: str):
    exist = False
    if os.path.isdir(f"scenarios/Scenario{scenario}"):
        exist = True
        for s in ["train", "val", "test"]:
            for id in range(5):
                if not os.path.isfile(
                    f"scenarios/Scenario{scenario}/{s}_gt_fold{id}.dat"
                ):
                    exist = False
                    break
    return exist


# Utility function for creating a random subset of the original train partition of Scenario X
def create_and_write_subset_train_folds(
    original_train_folds: List[str],
    p_size: float,
    scenario: str,
):
    for f in original_train_folds:
        with open(f, "r") as dat:
            samples = dat.read().splitlines()
        new_size = int(len(samples) * p_size / 100)
        new_samples = random.sample(samples, k=new_size)
        with open(f.replace("ScenarioX", f"Scenario{scenario}"), "w") as new_dat:
            new_dat.write("\n".join(new_samples))


# Utility function for creating 5-folds with train,
# validation, and test partitions for Scenarios 1, 4 and 7
# Validation and test partitions are those of Scenario X
# Train is a random subset of the original train partition of Scenario X
def create_folds_with_train_subset(p_size: float, scenario: str):
    # Check if the scenario already exists
    if not check_scenario_exists(scenario):
        # Obtain folds for ScenarioX
        folds = get_folds_filenames("X")
        # Create Scenario{scenario} folder
        os.makedirs(f"scenarios/Scenario{scenario}", exist_ok=True)
        # Copy val and test folds
        for f in folds["val"] + folds["test"]:
            shutil.copyfile(f, f.replace("ScenarioX", f"Scenario{scenario}"))
        # Create new train folds
        create_and_write_subset_train_folds(folds["train"], p_size, scenario)
    else:
        print(f"Scenario {scenario} already exists! Using existing folds.")
        pass


##################################### SCENARIOS 2, 3, 5, 6, 8, AND 9


# Utility function for writing 5-folds with train,
# validation, and test partitions for Scenario 2, 3, 5, 6, 8, and 9
# Train is a random subset of the original train partition of Scenario X
# Validation partitions are those of Sceneario X
# Test samples are passed as argument
def write_folds(test_samples: Dict[int, List[str]], train_p_size: float, scenario: str):
    # Obtain folds for ScenarioX
    folds = get_folds_filenames("X")
    # Create Scenario{scenario} folder
    os.makedirs(f"scenarios/Scenario{scenario}", exist_ok=True)
    # Create new train folds
    create_and_write_subset_train_folds(folds["train"], train_p_size, scenario)
    # Copy val folds
    for f in folds["val"]:
        shutil.copyfile(f, f.replace("ScenarioX", f"Scenario{scenario}"))
    # Create new test folds
    for id, samples in test_samples.items():
        with open(f"scenarios/Scenario{scenario}/test_gt_fold{id}.dat", "w") as dat:
            dat.write("\n".join(samples))

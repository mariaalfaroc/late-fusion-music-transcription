import os
import random

import numpy as np
import tensorflow as tf

from scenarios.folds_creation import (
    create_folds_with_train_subset,
    create_folds_with_train_and_test_subset,
)
from experimentation import k_fold_experiment, k_fold_test_experiment


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.config.list_physical_devices("GPU")

# Seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

if __name__ == "__main__":
    EPOCHS = 150
    BATCH_SIZE = {"omr": 16, "amt": 4}
    SCENARIOS_TRAIN_PSIZE_OMR = {
        "1": 2.5,
        "2": 2.2,
        "3": 2.0,
        "4": 4.0,
        "5": 3.7,
        "6": 3.0,
        "7": 10.5,
        "8": 9.0,
        "9": 6.3,
    }

    ##################################### EVALUATION ON SCENARIO X (ORIGINAL PARTITIONS) AND SCENARIOS FOLD CREATION:

    # 1) SCENARIO X
    # Evaluate AMT on Scenario X to be able to create test partitions based on model performance
    # There is no need to evaluate OMR on Scenario X as it will never be used for late multimodal fusion
    for task in ["omr", "amt"]:
        if task == "omr":
            continue
        k_fold_experiment(
            task=task,
            scenario_name="X",
            epochs=EPOCHS,
            batch_size=BATCH_SIZE[task],
        )

    # 2) SCENARIOS FOLD CREATION
    # Create folds for the rest of the scenarios to evaluate OMR
    for s, p_size in SCENARIOS_TRAIN_PSIZE_OMR.items():
        if s in ["1", "4", "7"]:
            create_folds_with_train_subset(p_size=p_size, scenario=s)
        elif s in ["2", "5", "8"]:
            create_folds_with_train_and_test_subset(
                train_p_size=p_size, scenario=s, symer_threshold=30
            )
        else:
            create_folds_with_train_and_test_subset(
                train_p_size=p_size, scenario=s, symer_threshold=10
            )

    ##################################### STAND-ALONE EVALUATION:
    for s in SCENARIOS_TRAIN_PSIZE_OMR.keys():
        # OMR must be trained and tested
        k_fold_experiment(
            task="omr",
            scenario_name=s,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE["omr"],
        )
        # AMT must be tested only
        if s in ["1", "4", "7"]:
            continue
        else:
            k_fold_test_experiment(task="amt", scenario_name=s)

    ##################################### KALDI PREPROCESSING::

    # TODO: JJ code should be placed here

    ##################################### MULTIMODAL EVALUATION:

    # TODO: Place multimodal calls here

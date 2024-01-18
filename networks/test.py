from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from networks.metrics import compute_metrics
from my_utils.data_processing import preprocess_input, preprocess_label


# CTC-greedy decoder:
# 1) Merge repeated elements
# 2) Remove blank labels
# 3) Convert back to string labels
def ctc_greedy_decoder(
    y_pred: tf.Tensor,
    input_length: List[int],
    i2w: Dict[int, str],
) -> List[List[str]]:
    input_length = tf.constant(input_length, dtype="int32", shape=(len(input_length),))
    # Blank labels are returned as -1
    y_pred = keras.backend.ctc_decode(y_pred, input_length, greedy=True)[0][0].numpy()
    # i2w conversion
    y_pred = [[i2w[int(i)] for i in b if int(i) != -1] for b in y_pred]
    return y_pred


# Utility function for evaluting a model over a dataset
# and computing the corresponding metrics
def evaluate_model(
    task: str,
    model: keras.Model,
    images_files: List[str],
    labels_files: List[str],
    i2w: Dict[int, str],
    print_metrics: bool = True,
) -> Tuple[float, float, Tuple[List[List[str]], List[np.ndarray], List[int]]]:
    y_true_acc = []
    raw_y_pred_acc = []
    y_pred_acc = []
    y_pred_len_acc = []
    # Iterate over images
    for img, label in zip(images_files, labels_files):
        images, images_len = preprocess_input(task=task, input_path=img)
        images = np.expand_dims(images, axis=0)
        # Obtain predictions
        y_pred = model(images, training=False)
        # Append raw predictions and input lengths to accumulator variables to later save them
        raw_y_pred_acc.extend(y_pred.numpy())
        y_pred_len_acc.append(images_len)
        # CTC greedy decoder (merge repeated, remove blanks, and i2w conversion)
        y_pred_acc.extend(ctc_greedy_decoder(y_pred, [images_len], i2w))
        # Obtain true labels
        y_true_acc.append(preprocess_label(label, training=False, w2i=None))

    # Compute metrics
    symer, seqer = compute_metrics(y_true_acc, y_pred_acc)
    if print_metrics:
        print(
            f"SymER (%): {symer:.2f}, SeqER (%): {seqer:.2f} - From {len(y_true_acc)} samples"
        )
    return symer, seqer, [y_true_acc, raw_y_pred_acc, y_pred_len_acc]

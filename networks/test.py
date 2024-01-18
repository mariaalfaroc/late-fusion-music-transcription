# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from data_processing import preprocess_image, preprocess_label


# CTC-greedy decoder (merge repeated elements, remove blank labels, and covert back to string labels)
def ctc_greedy_decoder(y_pred: tf.Tensor, input_length: list, i2w: dict) -> list:
    input_length = tf.constant(input_length, dtype="int32", shape=(len(input_length),))
    # Blank labels are returned as -1
    y_pred = keras.backend.ctc_decode(y_pred, input_length, greedy=True)[0][0].numpy()
    # i2w conversion
    y_pred = [[i2w[int(i)] for i in b if int(i) != -1] for b in y_pred]
    return y_pred


# --------------------


# --------------------


# Utility function for evaluting a model over a dataset and computing the corresponding metrics
def evaluate_model(model, images_files, labels_files, i2w):
    raw_y_pred_acc = []
    y_pred_acc = []
    y_pred_len_acc = []
    # Iterate over images in batches
    for start in range(0, len(images_files), config.batch_size):
        images, images_len = list(
            zip(
                *[
                    preprocess_image(i)
                    for i in images_files[start : start + config.batch_size]
                ]
            )
        )
        # Zero-pad images to maximum batch image width
        max_width = max(images, key=np.shape).shape[1]
        images = np.array(
            [
                np.pad(i, pad_width=((0, 0), (0, max_width - i.shape[1]), (0, 0)))
                for i in images
            ],
            dtype="float32",
        )
        # Obtain predictions
        y_pred = model(images, training=False)
        # Append raw predictions and input lengths to accumulator variables to later save them
        raw_y_pred_acc.extend(y_pred.numpy())
        y_pred_len_acc.extend(images_len)
        # CTC greedy decoder (merge repeated, remove blanks, and i2w conversion)
        y_pred_acc.extend(ctc_greedy_decoder(y_pred, images_len, i2w))
    # Obtain true labels
    y_true_acc = [preprocess_label(i, training=False, w2i=None) for i in labels_files]
    # Compute metrics
    symer, seqer = compute_metrics(y_true_acc, y_pred_acc)
    print(
        f"SymER (%): {symer:.2f}, SeqER (%): {seqer:.2f} - From {len(y_true_acc)} samples"
    )
    return symer, seqer, [y_true_acc, raw_y_pred_acc, y_pred_len_acc]

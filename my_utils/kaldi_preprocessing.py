# -*- coding: utf-8 -*-

import shutil, pathlib

import numpy as np

from my_utils.data_processing import load_dictionaries


def kaldi_fold(fold_filepath, kaldi_dir, fold_type):
    dst = kaldi_dir / f"ID_{fold_type}.lst"
    shutil.copyfile(src=fold_filepath, dst=dst)
    return


def kaldi_vocabulary(w2i_filepath, kaldi_dir):
    w2i, _ = load_dictionaries(w2i_filepath)
    with open(kaldi_dir / "chars.lst", "w") as f:
        for w in w2i.keys():
            f.write(f"{w}\n")
    return


def kaldi_groundtruth(kaldi_dir, id_labels, labels):
    with open(kaldi_dir / "grnTruth.dat", "w") as f:
        for id, y_true in zip(id_labels, labels):
            id = pathlib.Path(id)
            f.write(" ".join([str(id.stem)] + y_true) + "\n")
    return


def kaldi_confmat(kaldi_dir, fold_type, id_preds, preds, preds_len):
    with open(kaldi_dir / f"confMat-{fold_type}", "w") as f:
        for id, y_pred, len in zip(id_preds, preds, preds_len):
            id = pathlib.Path(id)
            f.write(" ".join([str(id.stem)] + ["[\n"]))
            for ts in np.log(y_pred[:len]):
                f.write(" ".join([str(c) for c in ts]) + "\n")
            f.write("]\n")
    return


def parse_kaldi_groundtruth(filepath):
    # This function can also be used to parse the results of the Kaldi-combination of word graphs
    gt = dict()
    lines = open(filepath).readlines()
    for line in lines:
        data = line.split()
        gt[data[0]] = data[1:]
    # Patch for ground-truth data test partition of Fold1
    if "201009318-1,48_2" in gt.keys():
        gt["201009318-1_48_2"] = gt["201009318-1,48_2"]
        del gt["201009318-1,48_2"]
    return gt

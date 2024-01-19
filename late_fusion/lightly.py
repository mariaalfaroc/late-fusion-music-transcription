import os, gc

import pandas as pd
import numpy as np

import config
from networks.test import compute_metrics
from my_utils.kaldi_preprocessing import parse_kaldi_groundtruth


# Utility function for performing a k-fold cross-validation multimodal experiment on a single dataset (LIGHT WORD-GRAPHS)
def k_fold_light_multimodal_experiment():
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print(
        "k-fold multimodal image and audio music transcription using light kaldi-combined word graphs experiment"
    )
    print(f"Data used {config.base_dir.stem}")

    # ---------- K-FOLD EVALUATION

    cwg_dir = config.output_dir / "LightWG_Decoded"

    # Start the k-fold evaluation scheme
    k = len(os.listdir(cwg_dir))
    for i in range(k):
        gc.collect()

        print(f"Fold {i}")

        # Load the current ground-truth data
        # Same file for both models (ofc), so load one of them
        kaldi_gt_path = (
            config.output_dir / "omr" / f"Fold{i}" / "kaldi" / "grnTruth.dat"
        )
        gt = parse_kaldi_groundtruth(filepath=kaldi_gt_path)

        # Set log filepath
        log_path = cwg_dir / f"Fold{i}" / "logs.csv"

        symer_acc = []
        seqer_acc = []
        for folder in ["Light_OMR_AMT", "Light_AMT_OMR"]:
            # Obtain predictions
            preds_path = cwg_dir / f"Fold{i}" / folder / "Results" / "Predictions.txt"
            preds = parse_kaldi_groundtruth(filepath=preds_path)
            assert gt.keys() == preds.keys()
            # Make sure they are in the same order
            y_true_acc = []
            y_pred_acc = []
            for k in gt.keys():
                y_true_acc.append(gt[k])
                y_pred_acc.append(preds[k])
            # Compute metrics
            symer, seqer = compute_metrics(y_true_acc, y_pred_acc)
            print(folder)
            print(
                f"SymER (%): {symer:.2f}, SeqER (%): {seqer:.2f} - From {len(y_true_acc)} samples"
            )
            symer_acc.append(symer)
            seqer_acc.append(seqer)
        # Save fold logs
        logs = {"symer": symer_acc, "seqer": seqer_acc}
        logs = pd.DataFrame.from_dict(logs)
        logs = logs.rename(index={0: "Light_OMR_AMT", 1: "Light_AMT_OMR"})
        logs.to_csv(log_path, index=True)

    return

# -*- coding: utf-8 -*-

import os, gc

import pandas as pd
import numpy as np

import config
from evaluation import compute_metrics
from kaldi_preprocessing import parse_kaldi_groundtruth

# Utility function for performing a k-fold cross-validation baseline experiment on a single dataset
def k_fold_baseline_experiment():
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print("k-fold cross-validation experiment (greedy decoded kaldi word graphs)")
    print(f"Data used {config.base_dir.stem}")

    # ---------- K-FOLD EVALUATION

    cwg_dir = config.output_dir / "CombinedWG_Decoded"

    # Start the k-fold evaluation scheme
    k = len(os.listdir(cwg_dir))
    for i in range(k):
        gc.collect()

        print(f"Fold {i}")

        # Load the current ground-truth data
        # Same file for both models (ofc), so load one of them
        kaldi_gt_path = config.output_dir / "omr" / f"Fold{i}" / "kaldi" / "grnTruth.dat"
        gt = parse_kaldi_groundtruth(filepath=kaldi_gt_path)

        # Set log filepath
        log_path = cwg_dir / f"Fold{i}" / "baseline_logs.csv"

        # Obtain predictions
        omr_preds_path = cwg_dir / f"Fold{i}" / "OMR10_AMT0" / "Results" / "greedy.txt"
        omr_preds = parse_kaldi_groundtruth(filepath=omr_preds_path)
        amt_preds_path = cwg_dir / f"Fold{i}" / "OMR0_AMT10" / "Results" / "greedy.txt"
        amt_preds = parse_kaldi_groundtruth(filepath=amt_preds_path)
        assert gt.keys() == omr_preds.keys() == amt_preds.keys()
        # Make sure they are in the same order
        y_true_acc = []
        omr_y_pred_acc = []
        amt_y_pred_acc = []
        for k in gt.keys():
            y_true_acc.append(gt[k])
            omr_y_pred_acc.append(omr_preds[k])
            amt_y_pred_acc.append(amt_preds[k])
        # Compute metrics
        omr_symer, omr_seqer = compute_metrics(y_true_acc, omr_y_pred_acc)
        amt_symer, amt_seqer = compute_metrics(y_true_acc, amt_y_pred_acc)
        print(f"OMR: SymER (%): {omr_symer:.2f}, SeqER (%): {omr_seqer:.2f} - From {len(y_true_acc)} samples")
        print(f"AMT: SymER (%): {amt_symer:.2f}, SeqER (%): {amt_seqer:.2f} - From {len(y_true_acc)} samples")
        # Save fold logs
        logs = {
            "omr_symer": [omr_symer],
            "omr_seqer": [omr_seqer],
            "amt_symer": [amt_symer],
            "amt_seqer": [amt_seqer],
        }
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(log_path, index=False)

    return

# Utility function for performing a k-fold cross-validation multimodal experiment on a single dataset
def k_fold_multimodal_experiment():
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print("k-fold multimodal image and audio music transcription using kaldi-combined word graphs experiment")
    print(f"Data used {config.base_dir.stem}")

    # ---------- K-FOLD EVALUATION

    cwg_dir = config.output_dir / "CombinedWG_Decoded"

    # Start the k-fold evaluation scheme
    k = len(os.listdir(cwg_dir))
    for i in range(k):
        gc.collect()

        print(f"Fold {i}")

        # Load the current ground-truth data
        # Same file for both models (ofc), so load one of them
        kaldi_gt_path = config.output_dir / "omr" / f"Fold{i}" / "kaldi" / "grnTruth.dat"
        gt = parse_kaldi_groundtruth(filepath=kaldi_gt_path)

        # Set log filepath
        log_path = cwg_dir / f"Fold{i}" / "logs.csv"

        symer_acc = []
        seqer_acc = []
        wfs = np.linspace(0,1,11)
        # Iterate over the range for the weight factor -> granularity step = 0.1
        for id, wf in enumerate(wfs):
            # Set weight factor
            if id == 0:
                wf_omr = 0
                wf_amt = 10
            elif id == 10:
                wf_omr = 10
                wf_amt = 0
            else:
                wf_omr = int(str(wf).split(".")[1][0])
                wf_amt = abs(10 - wf_omr)
            print(f"Weight factor for OMR: {wf_omr / 10}, Weight factor for AMT: {wf_amt / 10}")
            # Obtain predictions
            preds_path = cwg_dir / f"Fold{i}" / f"OMR{wf_omr}_AMT{wf_amt}" / "Results" / "Predictions.txt"
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
            print(f"SymER (%): {symer:.2f}, SeqER (%): {seqer:.2f} - From {len(y_true_acc)} samples")
            symer_acc.append(symer)
            seqer_acc.append(seqer)
        # Save fold logs
        logs = {
            "omr_weight_factor": [round(wf, 1) for wf in wfs],
            "amt_weight_factor": [round(1 - wf, 1) for wf in wfs],
            "symer": symer_acc,
            "seqer": seqer_acc
        }
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(log_path, index=False)

    return

# Utility function for performing a k-fold cross-validation multimodal experiment on a single dataset (LIGHT WORD-GRAPHS)
def k_fold_light_multimodal_experiment():
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print("k-fold multimodal image and audio music transcription using light kaldi-combined word graphs experiment")
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
        kaldi_gt_path = config.output_dir / "omr" / f"Fold{i}" / "kaldi" / "grnTruth.dat"
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
            print(f"SymER (%): {symer:.2f}, SeqER (%): {seqer:.2f} - From {len(y_true_acc)} samples")
            symer_acc.append(symer)
            seqer_acc.append(seqer)
        # Save fold logs
        logs = {
            "symer": symer_acc,
            "seqer": seqer_acc
        }
        logs = pd.DataFrame.from_dict(logs)
        logs = logs.rename(index={0: "Light_OMR_AMT", 1: "Light_AMT_OMR"})
        logs.to_csv(log_path, index=True)

    return

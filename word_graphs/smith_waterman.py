# -*- coding: utf-8 -*-

import os, gc

from typing import Tuple

import pandas as pd
import swalign

import config
from evaluation import compute_metrics
from kaldi_preprocessing import parse_kaldi_groundtruth

# Multimodal image and audio music transcription
# Carlos de la Fuente, Jose J. Valero-Mas, Francisco J. Castellanos, and Jorge Calvo-Zaragoza 

# OMR and AMT combination at prediction level:
# 1. Input the score image to the OMR model and the CQT image to the AMT model
# 2. Obtain predictions for each model
# 3. Apply alignment policy
    # 3.1. Greedy decoding policy
    # 3.2. Merge consecutive symbols (average)
    # 3.3. Remove CTC-blank symbols
    # 3.4. Smith-Waterman local alignment algorithm (swalign)
    # 3.5. Fusion policy
        # - Both sequences match on a token -> included
        # - Sequences disagree on a token -> include that of the highest probability
        # - A sequence misses a token -> include that of the other

# Steps 3.1 - 3.3 are performed using Kaldi

# --------------------

# Utility function for obtaining the predictions and their associated probability arrays out of a list of word graphs files
def get_predictions_and_probabilities(wg_files):
    y_pred_acc = dict()
    y_pred_prob_acc = dict()
    for f in wg_files:
        data = open(f).read().split("\n")
        id, data = data[0], data[1:]
        pred, prob = [], []
        for line in data:
            tokens = line.split()
            # To avoid empty lines
            if len(tokens) == 2:
                pred.append(tokens[0])
                prob.append(float(tokens[-1]))
        # Patch for ground-truth data test partition of Fold1 
        if id == "201009318-1,48_2":
            id = "201009318-1_48_2"
        y_pred_acc[id] = pred
        y_pred_prob_acc[id] = prob
    return y_pred_acc, y_pred_prob_acc

# --------------------

swalign_vocab = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
"$", "%", "&", "/", "(", ")", "=", "?", "¿", "*", "<", ">", "+", "#", "{", "}", ";", ":", "^", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Utility function for converting a string sequence to a compatible swalign string
def swalign_preprocess(r, q):
    current_vocab = sorted(set(r + q))
    # print(f"Length of current vocabulary: {len(current_vocab)} vs. Length of swalign vocabulary: {len(swalign_vocab)}")
    assert len(current_vocab) < len(swalign_vocab)
    w2swa = dict(zip(current_vocab, swalign_vocab))
    swa2w = dict(zip(swalign_vocab, current_vocab))
    r = ["¡"] + [w2swa[i] for i in r] + ["!"]
    q = ["¡"] + [w2swa[i] for i in q] + ["!"]
    return "".join(r), "".join(q), swa2w

# This is a modified version of the original dump() method for the Alignment class of the swalign library
# We have modified it to obtain (in the following order) the query, the matches, and the reference sequences; all of them have the same length
# Matches is a string that contains either "|" if sequences match on a token, or "." if they disagree, 
# or " " if one of them misses a token (in this case the token "-" is included at such position in the corresponding sequence)
def dump(alignment) -> Tuple[str, str, str]:
    i = alignment.r_pos
    j = alignment.q_pos

    q = ''
    m = ''
    r = ''
    qlen = 0
    rlen = 0

    for count, op in alignment.cigar:
        if op == 'M':
            qlen += count
            rlen += count
            for k in range(count):
                q += alignment.orig_query[j]
                r += alignment.orig_ref[i]
                if alignment.query[j] == alignment.ref[i] or (alignment.wildcard and (alignment.query[j] in alignment.wildcard or alignment.ref[i] in alignment.wildcard)):
                    m += '|'
                else:
                    m += '.'
                i += 1
                j += 1
        elif op == 'D':
            rlen += count
            for k in range(count):
                q += '-'
                r += alignment.orig_ref[i]
                m += ' '
                i += 1
        elif op == 'I':
            qlen += count
            for k in range(count):
                q += alignment.orig_query[j]
                r += '-'
                m += ' '
                j += 1
        elif op == 'N':
            q += '-//-'
            r += '-//-'
            m += '    '

    while q and r and m:
        return q, m, r

# Utility function for adapting the probility sequence after the swalign computation to be able to obtain the final alignment
def preprocess_prob(s: str, prob: list):
    new_prob = prob.copy()
    count = 0
    for id, v in enumerate(s):
        if v == "¡" or v == "!":
            new_prob.insert(id + count, 1)
            count += 1
        elif v == "-":
            new_prob.insert(id + count, 0)
            count += 1
    return new_prob

# Utility function for obtaining the final alignment sequence based on the fixed fusion policy
def get_alignment(q: str, m: str, r: str, q_prob: list, r_prob: list) -> str:
    alignment = ""
    for qv, mv, rv, qv_prob, rv_prob in zip(q, m, r, q_prob, r_prob):
        # There are three possible scenarios:
        # 1) Both sequences match on a token (mv == "|", qv == rv) -> included
        # 2) Sequences disagree on a token (mv == ".", qv != rv) -> include that of the highest probability
        # 3) A sequence misses a token (mv == " ", (qv or rv) == "-")-> include that of the other
        if mv == "|":
            # Scenario 1
            assert qv == rv
            alignment += qv
        elif mv == ".":
            # Scenario 2
            assert qv != rv
            alignment += qv if qv_prob >= rv_prob else rv
        elif mv == " ":
            # Scenario 3
            assert qv == "-" or rv == "-"
            alignment += qv if rv == "-" else rv
    return alignment

# Utility function for undoing the swalign preprocess
def undo_swalign_preprocess(alignment: str, swa2w: dict):
    return [swa2w[i] for i in list(alignment) if i not in ("¡", "!")]

# --------------------

# Utility function for evaluating a multimodal transcription at prediction level approach over a dataset
def evaluate_multimodal_transcription(omr_wg_files, amt_wg_files, y_true_acc_dict, match=2, mismatch=-1, gap_penalty=-1):
    y_true_acc = []
    y_pred_comb_acc = []
    # Obtain predictions and their corresponding probability arrays for OMR task
    omr_y_pred_acc, omr_y_pred_prob_acc = get_predictions_and_probabilities(omr_wg_files)
    # Obtain predictions and their corresponding probability arrays for AMT task
    amt_y_pred_acc, amt_y_pred_prob_acc = get_predictions_and_probabilities(amt_wg_files)
    # Obtain the callable object of swalign library that contains the align() method that performs the alignment
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    # Gap penalty designates scores for insertion or deletion
    sw =  swalign.LocalAlignment(scoring, gap_penalty)
    # Perform the multimodal combination at prediction level
    for (id_r, r), (id_r_prob, r_prob), (id_q, q), (id_q_prob, q_prob) in zip(omr_y_pred_acc.items(), omr_y_pred_prob_acc.items(), amt_y_pred_acc.items(), amt_y_pred_prob_acc.items()):
        assert id_r == id_r_prob == id_q == id_q_prob
        # Obtain true label
        y_true_acc.append(y_true_acc_dict[id_r])
        # Prepare for swalign computation
        r, q, swa2w = swalign_preprocess(r, q)
        # Smith-Waterman local alignment -> ref, query
        alignment = sw.align(r, q)
        q, m, r = dump(alignment)
        # Fusion policy
        q_prob = preprocess_prob(q, q_prob)
        r_prob = preprocess_prob(r, r_prob)
        alignment = get_alignment(q, m, r, q_prob, r_prob)
        # Undo the swalign preprocess and append to accumulator variable
        y_pred_comb_acc.append(undo_swalign_preprocess(alignment, swa2w))
    # Compute metrics
    symer, seqer = compute_metrics(y_true_acc, y_pred_comb_acc)
    print(f"SymER (%): {symer:.2f}, SeqER (%): {seqer:.2f} - From {len(y_true_acc)} samples")
    return symer, seqer

# --------------------

# Utility function for performing a k-fold cross-validation multimodal experiment on a single dataset
def k_fold_multimodal_experiment(match=[2], mismatch=[-1], gap_penalty=[-1]):
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print("k-fold multimodal image and audio music transcription combining kaldi word graphs following an Smith-Waterman policy experiment")
    print(f"Data used {config.base_dir.stem}")
    print(f"match={match}, mismatch={mismatch}, gap_penalty={gap_penalty}")

    # ---------- K-FOLD EVALUATION

    assert os.listdir(config.output_dir / "omr") == os.listdir(config.output_dir / "amt")

    # Start the k-fold evaluation scheme
    k = len(os.listdir(config.output_dir / "omr"))
    for i in range(k):
        gc.collect()

        print(f"Fold {i}")

        # Get the current fold data
        omr_wg_fnames = sorted([os.path.join(config.output_dir / "omr" / f"Fold{i}" / "SW", fname)  for fname in os.listdir(config.output_dir / "omr" / f"Fold{i}" / "SW")])
        amt_wg_fnames = sorted([os.path.join(config.output_dir / "amt" / f"Fold{i}" / "SW", fname)  for fname in os.listdir(config.output_dir / "amt" / f"Fold{i}" / "SW")])
        assert len(omr_wg_fnames) == len(amt_wg_fnames)

        # Load the current ground-truth data
        # Same file for both models (ofc), so load one of them
        kaldi_gt_path = config.output_dir / "omr" / f"Fold{i}" / "kaldi" / "grnTruth.dat"
        gt = parse_kaldi_groundtruth(filepath=kaldi_gt_path)

        # Set filepaths outputs
        output_dir = config.output_dir / "ResultsSW" / f"Fold{i}"
        os.makedirs(output_dir, exist_ok=True)
        log_path = output_dir / "logs.csv"

        symer_acc = []
        seqer_acc = []
        # Iterate over the different match, mismatch and gap_penalty values
        for m, mism, gp in zip(match, mismatch, gap_penalty):
            # Multimodal transcription evaluation
            symer, seqer = evaluate_multimodal_transcription(
                omr_wg_files=omr_wg_fnames, amt_wg_files=amt_wg_fnames, y_true_acc_dict=gt,
                match=m, mismatch=mism, gap_penalty=gp)
            symer_acc.append(symer)
            seqer_acc.append(seqer)
        # Save fold logs
        logs = {
            "match": match,
            "mismatch": mismatch,
            "gap_penalty": gap_penalty,
            "symer": symer_acc,
            "seqer": seqer_acc
        }
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(log_path, index=False)

    return

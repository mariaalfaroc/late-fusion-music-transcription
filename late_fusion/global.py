import os
import gc

import pandas as pd
import numpy as np

from networks.test import compute_metrics
from my_utils.kaldi_preprocessing import parse_kaldi_groundtruth

# Confusion Network Combination
# Combining Handwriting and Speech Recognition for Transcribing Historical Handwritten Documents
# Emilio Granell and Carlos-D. Martinez-Hinarejos

# A Confusion Network is a weighted directed graph, in which each path goes through all the nodes.
# The words and their probabilities are stored in the edges, and the total probability of the words
# contained in a subnetwork (all edges between two consecutive nodes) sum 1.
# Aspect of a Confusion Network:
# CNX -> Class Number X ; PCNX -> Probability of Class Number X
# ID_SAMPLE [ CN1 PCN1 ] [ CN1 PCN1 ] [ CN1 PCN1 CN2 PCN2 ] [ CN1 PCN1 ] [ CN1 PCN1 ]
# [ CN1 PCN1 ] -> This is a subnetwork
# Example: 000116002-1_1_1 [ 0 1 ] [ 12 1 ] [ 2 0.6030554 0 0.3969446 ] [ 0 1 ] [ 2 1 ]

# WORKFLOW
# ----
# FIRST PART -> Dynamic Time Warping (DTW) based alignment
# We obtain (i) the cost matrix, which indicates the Symbol Error Rate between all the possible combinations between a SN of the first CN and another SN of the second CN;
# and (ii) the best alignment path, which aligns each SN of the first CN with another SN of the second CN, so that the cost of this alignment is the lowest possible.
# ----
# SECOND PART -> Subnetworks based alignment
# We go through the best alignment path, SN-SN pair by SN-SN pair, classifying each of the subnetworks that form this pair as "Anchor", "Combine", or "Insert/Delete".
# Following the gram matching threshold policy of the paper:
# "Anchor" -> The Symbol Error Rate of the SN-SN pair is 0.
# "Combine" -> The Symbol Error Rate of the SN-SN pair is not 0, but the subnetworks have some elements in common (perform the intersection between them to know so)
# "Insert/Delete -> The Symbol Error Rate of the SN-SN pair is not 0 and the subnetworks have nothing in common (perform the intersection between them to know so)
# Tie breaking rule for insertion or deletion -> We look for the highest probability and follow the threshold policy of the paper.
# We perform this subnetwork classification in both directions of the best alignment path, from left to right and from right to left.
# We take only as "Anchor" subnetworks those where both searches coincide.
# Tie breaking rule for one-to-many matches: Anchor > Combine > Insert/Delete. That is if a subnetwork is classified as both "Anchor" and "Insert/Delete",
# we keep the label that minimizes the alignment cost -> "Anchor" label.
# ----
# THIRD PART -> Composition of the final confusion network
# We go through the best alignment path and insert first only "Anchor" subnetworks and then the rest.

# Kaldi extra vocabulary:
# <eps> -> Means "no symbol here". It is the "*DELETE*" of the paper.
# <DUMMY> -> Used by the HMM to create the WFST need for the posteriogram -> word graph coversion.
# #0 -> As in the previous case, due to a disambiguation issue at the level of one of the FSTs mounted by the Kaldi system.
# <s> -> Indicates the start of a sequence.
# </s> 1170 -> Indicates the end of a sequence.

# -------------------- FIRST PART


# Utility function for computing the cost (Symbol Error Rate or Intersection) between two subnetworks
def compute_cost(sn1, sn2, c_type: str):
    # Obtain the classes of each subnetwork
    sn1_syms, sn2_syms = sn1[0::2], sn2[0::2]
    if c_type == "ser":
        # cost == Symbol Error Rate (%)
        return compute_metrics([sn1_syms], [sn2_syms])[0]
    elif c_type == "instersection":
        # cost == Intersection
        return 0 if len(list(set(sn1_syms).intersection(set(sn2_syms)))) > 0 else 1


# Utility function for obtaining the cost matrix of two subnetworks and the best DTW path for aligning them together
# Source: https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd
def dtw(cn1, cn2):
    # DTW Matrix
    # matrix of len(cn1) + 1 x len(cn2) + 1 -> len(cnX) = number of subnetworks of the confusion network cnX
    len_cn1, len_cn2 = len(cn1), len(cn2)
    dtw_matrix = np.full((len_cn1 + 1, len_cn2 + 1), np.inf)
    dtw_matrix[0, 0] = 0
    # DTW Best Path Matrix (utility matrix for computing the best DTW path)
    path_matrix = np.zeros((len_cn1 + 1, len_cn2 + 1))
    # Cost matrix
    cost_matrix = np.zeros((len_cn1, len_cn2))
    for i in range(1, len_cn1 + 1):
        for j in range(1, len_cn2 + 1):
            # Compute current cost (Symbol Error Rate)
            cost = compute_cost(cn1[i - 1], cn2[j - 1], c_type="ser")
            cost_matrix[i - 1, j - 1] = cost
            # Compute the "minimum cost" to achieve the current cost
            neighbors = [
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            ]
            last_min = min(neighbors)
            dtw_matrix[i, j] = cost + last_min
            # Compute best neighbor
            id_last_min = neighbors.index(last_min)
            path_matrix[i, j] = id_last_min
    # DTW Best Path
    path = []
    i, j = len_cn1, len_cn2
    while not (i == j == 0):
        path.append((i - 1, j - 1))
        v = path_matrix[i, j]
        i = i if v == 1 else i - 1
        j = j if v == 0 else j - 1
    path.reverse()
    return cost_matrix, path


# -------------------- SECOND PART


# Utility function for performing tie breaking rule over subnetworks classified as "Insert/Delete"
def insertion_or_deletion_labeling(sn, modality):
    sn_type = ""
    insert_threshold = 0.25  # Most probable class of the second modality exceeds this threshold -> Insert
    delete_threshold = 0.75  # Most probable class of the first modality does not reach this threshold -> Delete
    # sn = [class-1 prob-1 class-2 prob-2] -> sn_dict = {"class-1": prob-1, "class-2": prob-2}
    sn_dict = dict(zip(sn[0::2], sn[1::2]))
    # Find the most probable class of the subnetwork
    most_prob_c = max(sn_dict.values())
    if modality == 1:
        if most_prob_c > delete_threshold:
            # Insert
            sn_type = "Insert"
        else:
            # Delete
            sn_type = "Delete"
    elif modality == 2:
        if most_prob_c > insert_threshold:
            # Insert
            sn_type = "Insert"
        else:
            # Delete
            sn_type = "Delete"
    return sn_type


# Utility function for labeling subnetworks as either "Anchor", or "Combine", or "Insert/Delete"
def subnetworks_labeling(cn1, cn2, cost_matrix, path):
    cn1_type = {"Anchor": [], "Combine": [], "Insert": [], "Delete": []}
    cn2_type = {"Anchor": [], "Combine": [], "Insert": [], "Delete": []}
    # We iterate over the DTW path
    # Ex.: i=3, j=4 -> This indicates that SN3 from CN1 aligns with SN4 from CN2
    for i, j in path:
        # Get the cost of aligning subnetworks i (from CN1) and j (from CN2)
        cost = cost_matrix[i, j]
        # Gram matching threshold -> threshold == 0
        if cost == 0:
            # The two subnetworks agree, i.e., their hypothesis match (we only take into account the classes and not their associated probabilities)
            # This means the cost (the value of the Symbol Error Rate) is 0
            # We classify them as "Anchor"
            sn1_type = sn2_type = "Anchor"
        else:
            # The two subnetworks disagree, i.e., their hypothesis do not match
            # They may be of type "Combine" or of type "Insert/Delete"
            # Let's see if their hypothesis are completely different or not
            # Compute the intersection between both subnetworks to know so
            value = compute_cost(cn1[i], cn2[j], c_type="instersection")
            if value == 0:
                # They partially disagree -> they have some common ground
                # We classify them as "Combine"
                sn1_type = sn2_type = "Combine"
            else:
                # They completely disagree
                # We classify them as "Insert" or "Delete"
                sn1_type, sn2_type = insertion_or_deletion_labeling(
                    cn1[i], modality=1
                ), insertion_or_deletion_labeling(cn2[j], modality=2)
        cn1_type[sn1_type].append(i)
        cn2_type[sn2_type].append(j)
    return cn1_type, cn2_type


# Utility function for reversing a "cn_type" dictionary
# Example:
# cn_type =  {"Anchor": [1, 2], "Combine": [3], "Insert": [4], "Delete": [5]}
# cn_type_reverse = {1: "Anchor", 2: "Anchor", 3: "Combine", 4: "Insert", 5: "Delete"}
def reverse_dict(cn_type):
    cn_type_reverse = dict()
    for k in cn_type.keys():
        for v in cn_type[k]:
            cn_type_reverse[int(v)] = k
    return cn_type_reverse


# Utility function for merging the labeling in both directions of a confusion network
def subnetworks_labeling_alignment(cn, cn_type_1, cn_type_2):
    # Ex.: cn_type_x =  {"Anchor": [1, 2], "Combine": [3], "Insert": [4], "Delete": [5]}
    types = {"Anchor": 100, "Combine": 50, "Insert": 0, "Delete": 0}
    # FIRST STEP
    # We take only as "Anchor" subnetworks those where both searches coincide
    cn_type = dict()
    for v in types.keys():
        cn_type[v] = set(cn_type_1[v]).intersection(set(cn_type_2[v]))
    # SECOND STEP
    # DTW aligns subnetworks in such a way that we may find one-to-many matches
    # Therefore, we might find subnetworks classified as more than one type
    for i in types.keys():
        for j in types.keys():
            if i != j:
                matches = set(cn_type[i]).intersection(set(cn_type[j]))
                if len(matches) > 0:
                    for sn in matches:
                        # print(f"Subnetwork {sn} classify as both {i} and {j}")
                        # Tie breaking rule: Anchor > Combine > Insert/Delete
                        # That is if a subnetwork is classified as both "Anchor" and "Insert/Delete",
                        # we keep the label that minimizes the alignment cost -> "Anchor" label
                        if types[i] > types[j]:
                            cn_type[j].remove(sn)
                            # print(f"Now, subnetwork {sn} is classified only as {i}")
                        else:
                            cn_type[i].remove(sn)
                            # print(f"Now, subnetwork {sn} is classified only as {j}")
    return reverse_dict(cn_type)


# Utility function for performing a subnetworks based alignment
def subnetworks_based_alignment(cn1, cn2, cost_matrix, path):
    # Label subnetworks in both directions
    cn1_type_1, cn2_type_1 = subnetworks_labeling(cn1, cn2, cost_matrix, path)
    cn1_type_2, cn2_type_2 = subnetworks_labeling(cn1, cn2, cost_matrix, path[::-1])
    # Unify the searches
    cn1_type = subnetworks_labeling_alignment(cn1, cn1_type_1, cn1_type_2)
    cn2_type = subnetworks_labeling_alignment(cn2, cn2_type_1, cn2_type_2)
    return cn1_type, cn2_type


# -------------------- THIRD PART


def set_weight_factor(value=0.5):
    global weight_factor
    weight_factor = round(value, 1)


# Utility function for computing the final probability of a class given their probabilities in other subnetworks (used when those subnetworks are combined)
def smooth_probability(prob_c_sn1, prob_c_sn2, n1, n2):
    granularity_factor = 10e-4
    prob_c_sn1 = (prob_c_sn1 + granularity_factor) / (1 + n1 * granularity_factor)
    prob_c_sn2 = (prob_c_sn2 + granularity_factor) / (1 + n2 * granularity_factor)
    # weight_factor = 0.5
    return pow(prob_c_sn1, weight_factor) * pow(prob_c_sn2, round(1 - weight_factor, 1))


# Utility function for combining two subnetworks: f(sn1, sn2) -> sn
def combine_subnetworks(sn1, sn2):
    # snx = [class-1 prob-1 class-2 prob-2] -> snx_dict = {"class-1": prob-1, "class-2": prob-2}
    sn1_dict = dict(zip(sn1[0::2], sn1[1::2]))
    sn2_dict = dict(zip(sn2[0::2], sn2[1::2]))
    # Obtain all the classes
    # If both sn1 and sn2 are of type "Anchor", they both have the same classes -> sn1_c == sn2_c == sn_c
    # If both sn1 and sn2 are of type "Combine" -> sn_c == sn1_c U sn2_c
    sn_c = set(sn1[0::2] + sn2[0::2])
    n1, n2 = len(sn1[0::2]), len(sn2[0::2])
    # Smooth and normalize the probabilities of the classes of the combined subnetwork
    probs = [
        smooth_probability(sn1_dict.get(c, 0), sn2_dict.get(c, 0), n1, n2) for c in sn_c
    ]
    probs_normalized = [p / sum(probs) for p in probs]
    # Create the combined subnetwork
    sn = []
    for c, prob_c in zip(sn_c, probs_normalized):
        sn.append(c)
        sn.append(prob_c)
    return sn


# Utility function for merging two aligned subnetworks into a new one
def confusion_networks_alignment(path, cn1, cn2, cn1_type, cn2_type):
    # Ex.: cnx_type = {1: "Anchor", 2: "Anchor", 3: "Combine", 4: "Insert", 5: "Delete"}
    # Ex.: path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)], then -> cn = [[[], []], [[], []], [[], []], [[], []], [[], []]]
    # We fill in the combined confusion network taking into account the pairs given by the best DTW path
    cn = [[[], []] for _ in path]

    # First, insert only "Anchor" pairs
    for id, (i, j) in enumerate(path):
        if cn1_type[int(i)] == cn2_type[int(j)] and cn1_type[int(i)] == "Anchor":
            sn1, sn2 = combine_subnetworks(cn1[i], cn2[j]), ["-"]
            cn[id][0], cn[id][1] = sn1, sn2

    # Second, insert the rest
    eps_sn = ["<eps>", 1.0]
    for id, (i, j) in enumerate(path):
        # We only fill in the empty gaps
        if cn[id][0] == [] and cn[id][1] == []:
            # Both subnetworks are of the same type
            # Combine-Combine, Insert-Insert, Delete-Delete
            if cn1_type[int(i)] == cn2_type[int(j)]:
                if cn1_type[int(i)] == "Combine":
                    sn1, sn2 = combine_subnetworks(cn1[i], cn2[j]), ["-"]
                elif cn1_type[int(i)] == "Insert" or cn1_type[int(i)] == "Delete":
                    sn1, sn2 = combine_subnetworks(cn1[i], eps_sn), combine_subnetworks(
                        eps_sn, cn2[j]
                    )
            # Subnetworks are of different type
            # Anchor-Combine, Anchor-Insert, Anchor-Delete, Combine-Insert, Combine-Delete, Insert-Delete
            else:
                # Anchor-Combine, Anchor-Insert, Anchor-Delete
                if cn1_type[int(i)] == "Anchor" or cn2_type[int(j)] == "Anchor":
                    # The "Anchor" subnetwork has been inserted already
                    if cn1_type[int(i)] == "Anchor":
                        sn1 = ["-"]
                        sn2 = (
                            combine_subnetworks(cn1[i], cn2[j])
                            if cn2_type[int(j)] == "Combine"
                            else combine_subnetworks(eps_sn, cn2[j])
                        )
                    elif cn2_type[int(j)] == "Anchor":
                        sn1 = (
                            combine_subnetworks(cn1[i], cn2[j])
                            if cn1_type[int(i)] == "Combine"
                            else combine_subnetworks(cn1[i], eps_sn)
                        )
                        sn2 = ["-"]
                # Combine-Insert, Combine-Delete, Insert-Delete
                else:
                    # NOTE: Assumption! -> When we find a Combine-Insert or a Combine-Delete pair, the Combine subnetwork has another possible combination
                    sn1 = (
                        ["-"]
                        if cn1_type[int(i)] == "Combine"
                        else combine_subnetworks(cn1[i], eps_sn)
                    )
                    sn2 = (
                        ["-"]
                        if cn2_type[int(j)] == "Combine"
                        else combine_subnetworks(eps_sn, cn2[j])
                    )
            cn[id][0], cn[id][1] = sn1, sn2

    # Third, filter the dummy values -> sn = ["-"] and reformat the structure
    for pair in cn:
        if ["-"] in pair:
            pair.remove(["-"])
    cn = [sn for pair in cn for sn in pair]

    return cn


# Utility function for following the workflow established to combine two confusion networks
def combine_confusion_networks(cn1, cn2):
    # FIRST PART
    # DTW based alignment
    cost_matrix, path = dtw(cn1, cn2)
    # SECOND PART
    # Subnetworks based alignment
    # Ex.: cnx_type = {1: "Anchor", 2: "Anchor", 3: "Combine", 4: "Insert", 5: "Delete"}
    cn1_type, cn2_type = subnetworks_based_alignment(cn1, cn2, cost_matrix, path)
    # THIRD PART
    # Composition of the final confusion network
    cn = confusion_networks_alignment(path, cn1, cn2, cn1_type, cn2_type)
    return cn


# -------------------- UTILS


# Utility function for parsing a confusion network file
# Returns: id = "ID_SAMPLE", out=[[CN1, PCN1], [CN1, PCN1], [CN1, PCN1, CN2, PCN2], [CN1, PCN1], [CN1, PCN1]]
def confnet_str2list(v: list):
    id, confnet = v[0], v[1:]
    # Patch for ground-truth data test partition of Fold1
    if id == "201009318-1,48_2":
        id = "201009318-1_48_2"
    out = []
    for i in confnet:
        if i == "[":
            # We enter a node
            node = []
        elif i == "]":
            # Append the node
            out.append(node)
        else:
            # We are inside a node
            i = float(i) if len(i.split(".")) > 1 else int(i)
            node.append(i)
    return id, out


# Utility function for doing the i2w-conversion over the classes of a confusion network
def confnet_i2w(cn: list, i2w: dict):
    out = []
    for sn in cn:
        new_sn = []
        for id, sym in enumerate(sn):
            if id % 2 == 0:
                new_sn.append(i2w[sym])
            else:
                new_sn.append(sym)
        out.append(new_sn)
    return out


# Utility function for greedy decoding a subnetwork
def confnet_greedy_decoder(cn):
    cn_decoded = []
    for sn in cn:
        classes, probs = sn[0::2], sn[1::2]
        max_prob = max(probs)
        # See if there is more than one class with the same probability
        ids_max_prob = [id for id, p in enumerate(probs) if p == max_prob]
        most_prob_c = [
            classes[i]
            for i in ids_max_prob
            if classes[i] not in ["<eps>", "<DUMMY>", "#0"]
        ]
        if most_prob_c != []:
            # We append the first word we find different from "<eps>"
            cn_decoded.append(most_prob_c[0])
    return cn_decoded


# -------------------- EXPERIMENT WORKFLOW


# Utility function for performing a k-fold cross-validation multimodal experiment on a single dataset
def k_fold_multimodal_experiment():
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print(
        "k-fold multimodal image and audio music transcription using confusion networks experiment"
    )
    print(f"Data used {config.base_dir.stem}")

    # ---------- K-FOLD EVALUATION

    assert os.listdir(config.output_dir / "omr") == os.listdir(
        config.output_dir / "amt"
    )

    # Start the k-fold evaluation scheme
    k = len(os.listdir(config.output_dir / "omr"))
    for i in range(k):
        gc.collect()

        print(f"Fold {i}")

        # Get the current fold data
        omr_confnets_fnames = sorted(
            [
                os.path.join(config.output_dir / "omr" / f"Fold{i}" / "CN", fname)
                for fname in os.listdir(config.output_dir / "omr" / f"Fold{i}" / "CN")
            ]
        )
        amt_confnets_fnames = sorted(
            [
                os.path.join(config.output_dir / "amt" / f"Fold{i}" / "CN", fname)
                for fname in os.listdir(config.output_dir / "amt" / f"Fold{i}" / "CN")
            ]
        )
        assert len(omr_confnets_fnames) == len(amt_confnets_fnames)

        # Load the current ground-truth data
        # Same file for both models (ofc), so load one of them
        kaldi_gt_path = (
            config.output_dir / "omr" / f"Fold{i}" / "kaldi" / "grnTruth.dat"
        )
        gt = parse_kaldi_groundtruth(filepath=kaldi_gt_path)

        # Load the current fold dictionary
        # Both models have the same fold vocabulary, so load one of them
        w2i_filepath = config.output_dir / "omr" / f"Fold{i}" / "words.txt"
        lines = open(w2i_filepath, "r").readlines()
        i2w = {int(line.split()[1]): line.split()[0] for line in lines}

        # Set filepaths outputs
        output_dir = config.output_dir / "ResultsCN" / f"Fold{i}"
        os.makedirs(output_dir, exist_ok=True)
        log_path = output_dir / "logs.csv"

        symer_acc = []
        seqer_acc = []
        wfs = np.linspace(0, 1, 11)
        # Iterate over the range for the weight factor -> granularity step = 0.1
        for wf in wfs:
            # Set weight factor
            set_weight_factor(wf)
            print(
                f"Weight factor for OMR: {weight_factor}, Weight factor for AMT: {round(1 - weight_factor, 1)}"
            )
            # Multimodal transcription evaluation
            labels_files = []
            y_pred_acc = []
            for omr_confnet_filepath, amt_confnet_filepath in zip(
                omr_confnets_fnames, amt_confnets_fnames
            ):
                id_omr, omr_confnet = confnet_str2list(
                    open(omr_confnet_filepath, "r").read().split()
                )
                id_amt, amt_confnet = confnet_str2list(
                    open(amt_confnet_filepath, "r").read().split()
                )
                assert id_omr == id_amt
                labels_files.append(id_omr)
                combined_confnet = combine_confusion_networks(
                    confnet_i2w(omr_confnet, i2w), confnet_i2w(amt_confnet, i2w)
                )
                y_pred_acc.append(confnet_greedy_decoder(combined_confnet))
            # Obtain true labels: we make sure they are in the same order as their correspoding y_pred partner
            y_true_acc = [gt[i] for i in labels_files]
            # Compute metrics
            symer, seqer = compute_metrics(y_true_acc, y_pred_acc)
            print(
                f"SymER (%): {symer:.2f}, SeqER (%): {seqer:.2f} - From {len(y_true_acc)} samples"
            )
            symer_acc.append(symer)
            seqer_acc.append(seqer)
        # Save fold logs
        logs = {
            "omr_weight_factor": [round(wf, 1) for wf in wfs],
            "amt_weight_factor": [round(1 - wf, 1) for wf in wfs],
            "symer": symer_acc,
            "seqer": seqer_acc,
        }
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(log_path, index=False)

    return


# if __name__ == "__main__":
#     cn1 = [["<s>", 1.0], ["A", 0.559, "E", 0.294, "<eps>", 0.147], ["AGORA", 1.0], ["CUENTA", 1.0], ["LABRADORES", 0.918, "LA", 0.082], ["</s>", 1.0]]
#     cn2 = [["<s>", 1.0], ["AGORA", 1.0], ["CUENTA", 1.0], ["EL", 0.657, "LA", 0.343], ["HISTORIA", 1.0], ["</s>", 1.0]]
#     set_weight_factor()
#     cn = combine_confusion_networks(cn1, cn2)
#     print(confnet_greedy_decoder(cn))

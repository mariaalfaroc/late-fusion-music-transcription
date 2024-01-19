from cProfile import run
import os
import shutil
from editdistance import distance


path_to_kaldi = "~/Tools/kaldi"
lattice_best_path = os.path.join(path_to_kaldisrc, "latbin/lattice-best-path")
lattice_combine_path = os.path.join(path_to_kaldi, "src/latbin/lattice-combine")
lattice_combine_light_path = os.path.join(
    path_to_kaldi, "src/latbin/lattice-combine-light"
)  # https://github.com/jfainberg/lattice_combination
lattice_copy = os.path.join(path_to_kaldi, "src/latbin/lattice-copy")
lattice_mbr = os.path.join(path_to_kaldi, "src/latbin/lattice-mbr-decode")
int2sym_path = "utils/int2sym.pl"


def CombineWordGraphs(src_OMR, src_AMT, WG_dst, words_path):
    # Listing files:
    files = [f for f in os.listdir(src_OMR) if f.endswith(".lat")]

    # Weights vector:
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for current_weight in weights:
        base_dst_path = os.path.join(
            WG_dst,
            "OMR{}_AMT{}".format(
                int(10 * current_weight), 10 - int(10 * current_weight)
            ),
        )

        if not os.path.exists(base_dst_path):
            os.makedirs(base_dst_path)
        WG_dst_path = os.path.join(base_dst_path, "WG")
        if not os.path.exists(WG_dst_path):
            os.makedirs(WG_dst_path)
        Results_dst_path = os.path.join(base_dst_path, "Results")
        if not os.path.exists(Results_dst_path):
            os.makedirs(Results_dst_path)

        fout = open(os.path.join(Results_dst_path, "Predictions.txt"), "w")
        lat_weights = "--lat-weights='{:.1f}:{:.1f}'".format(
            current_weight, 1 - current_weight
        )
        for single_file in files:
            # Merging WGs:
            OMR_file = "ark,t:" + os.path.join(src_OMR, single_file) + " "
            AMT_file = "ark,t:" + os.path.join(src_AMT, single_file) + " "
            mixture = (
                lattice_combine_path
                + " "
                + lat_weights
                + " "
                + OMR_file
                + " "
                + AMT_file
                + " ark,t:- > "
                + os.path.join(WG_dst_path, single_file.split(".")[0] + ".lat")
            )
            os.system(mixture)

            # Decoding merged WGs:
            decoding = (
                lattice_best_path
                + " ark,t:- ark,t:-|"
                + int2sym_path
                + " -f 2- "
                + words_path
            )
            line_call = (
                lattice_combine_path
                + " "
                + lat_weights
                + " "
                + OMR_file
                + " "
                + AMT_file
                + " ark,t:-|"
                + decoding
                + " > temp.txt"
            )

            os.system(line_call)

            with open("temp.txt") as f:
                FileRead = f.readlines()[0].strip()

            fout.write(FileRead + "\n")

        fout.close()

    os.remove("temp.txt")

    return


def ExtremeCases(src_AMT, src_OMR, WG_dst, words_path):
    dict_exps = {"AMT": src_AMT, "OMR": src_OMR}
    dict_exps = {"OMR": src_OMR}

    for modality, path_modality in dict_exps.items():
        files = [f for f in os.listdir(path_modality) if f.endswith(".gz")]

        base_dst_path = os.path.join(WG_dst, modality)
        if not os.path.exists(base_dst_path):
            os.makedirs(base_dst_path)

        lattice_dst_path = os.path.join(base_dst_path, "WG")
        if not os.path.exists(lattice_dst_path):
            os.makedirs(lattice_dst_path)

        prediction_dst_path = os.path.join(base_dst_path, "Results")
        if not os.path.exists(prediction_dst_path):
            os.makedirs(prediction_dst_path)

        fout = open(os.path.join(prediction_dst_path, "Predictions.txt"), "w")
        for single_file in files:
            # Lattice:
            current_file = (
                "'ark:gzip -c -d " + os.path.join(path_modality, single_file) + "|'"
            )
            line_call = (
                lattice_copy
                + " "
                + current_file
                + " ark,t:-> "
                + os.path.join(lattice_dst_path, single_file.split(".")[0] + ".lat")
            )
            os.system(line_call)

            # Prediction:
            decoding_call = (
                lattice_best_path
                + " "
                + current_file
                + " ark,t:-| "
                + int2sym_path
                + " -f 2- "
                + words_path
                + " > temp.txt"
            )

            os.system(decoding_call)
            with open("temp.txt") as f:
                FileRead = f.readlines()[0].strip()

            fout.write(FileRead + "\n")

        fout.close()

    os.remove("temp.txt")
    return


def CombineLightWordGraphs(src_OMR, src_AMT, dst_folder_name, words_path):
    # Listing files:
    files = [f for f in os.listdir(src_OMR) if f.endswith(".lat")]

    base_dst_path = "/".join(src_OMR.split("/")[:2])

    dst_path = os.path.join(base_dst_path, dst_folder_name)

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    WG_dst_path = os.path.join(dst_path, "WG")
    if not os.path.exists(WG_dst_path):
        os.makedirs(WG_dst_path)
    Results_dst_path = os.path.join(dst_path, "Results")
    if not os.path.exists(Results_dst_path):
        os.makedirs(Results_dst_path)

    fout = open(os.path.join(Results_dst_path, "Predictions.txt"), "w")
    for single_file in files:
        # Merging WGs:
        OMR_file = "ark,t:" + os.path.join(src_OMR, single_file)
        AMT_file = "ark,t:" + os.path.join(src_AMT, single_file)
        mixture = (
            lattice_combine_light_path
            + " "
            + OMR_file
            + " "
            + AMT_file
            + " ark,t:- > "
            + os.path.join(WG_dst_path, single_file.split(".")[0] + ".lat")
        )
        os.system(mixture)

        # Decoding merged WGs:
        decoding = (
            lattice_best_path
            + " ark,t:- ark,t:-|"
            + int2sym_path
            + " -f 2- "
            + words_path
        )
        line_call = (
            lattice_combine_light_path
            + " "
            + OMR_file
            + " "
            + AMT_file
            + " ark,t:-|"
            + decoding
            + " > temp.txt"
        )

        os.system(line_call)

        with open("temp.txt") as f:
            FileRead = f.readlines()[0].strip()

        fout.write(FileRead + "\n")

    fout.close()

    return


def CombineMBRFilesForSWProbs():
    with open("temp1.txt") as fin:
        tokens = fin.readlines()[0].split()

    with open("temp2.txt") as fin:
        probs = [
            " ".join(u.split("]")).strip().split()
            for u in fin.readlines()[0].strip().split("[")[1:]
        ]

    # Processing probs:
    probs = [u[:2] for u in probs if u[0] != "0"]

    file_name = tokens[0]
    tokens = tokens[1:]

    for it in range(len(probs)):
        probs[it][0] = tokens[it]

    return file_name, probs


def ObtainSWProbs(src_AMT, src_OMR, words_path):
    # Listing files:
    files = [f for f in os.listdir(src_OMR) if f.endswith(".lat")]

    for single_file in files:
        # OMR CASE:
        base_dst_path = "/".join(src_OMR.split("/")[:-1])
        dst_path = os.path.join(base_dst_path, "SW")
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        OMR_file = "ark,t:" + os.path.join(src_OMR, single_file)

        line = (
            lattice_mbr
            + " "
            + OMR_file
            + " 'ark,t:|"
            + int2sym_path
            + " -f 2- "
            + words_path
            + " > temp1.txt'"
            + " ark:/dev/null ark,t:temp2.txt"
        )
        os.system(line)
        name, probs = CombineMBRFilesForSWProbs()

        with open(os.path.join(dst_path, name), "w") as fout:
            fout.write(name + "\n")
            for line in probs:
                fout.write(" ".join(line) + "\n")

        # AMT CASE:
        base_dst_path = "/".join(src_AMT.split("/")[:-1])
        dst_path = os.path.join(base_dst_path, "SW")
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        AMT_file = "ark,t:" + os.path.join(src_AMT, single_file)

        line = (
            lattice_mbr
            + " "
            + AMT_file
            + " 'ark,t:|"
            + int2sym_path
            + " -f 2- "
            + words_path
            + " > temp1.txt'"
            + " ark:/dev/null ark,t:temp2.txt"
        )
        os.system(line)
        name, probs = CombineMBRFilesForSWProbs()

        with open(os.path.join(dst_path, name), "w") as fout:
            fout.write(name + "\n")
            for line in probs:
                fout.write(" ".join(line) + "\n")

    os.remove("temp1.txt")
    os.remove("temp2.txt")
    return


def ConfusionNetworks(src_OMR, src_AMT, WG_dst):
    if os.path.exists(os.path.join(WG_dst, "OMR", "CN")):
        shutil.rmtree(os.path.join(WG_dst, "OMR", "CN"))
    shutil.copytree(src_OMR, os.path.join(WG_dst, "OMR", "CN"))

    if os.path.exists(os.path.join(WG_dst, "AMT", "CN")):
        shutil.rmtree(os.path.join(WG_dst, "AMT", "CN"))
    shutil.copytree(src_AMT, os.path.join(WG_dst, "AMT", "CN"))

    return


def evaluate_dictionaries(gt_dict, prediction_dict):
    keys = list(gt_dict.keys())

    running_length = 0
    errors = 0

    for single_key in keys:
        errors += distance(gt_dict[single_key], prediction_dict[single_key])
        running_length += len(gt_dict[single_key])

    return errors, running_length


def results_summary(base_path):
    fresults = open(os.path.join(base_path, "Results_summary.txt"), "w")

    with open(os.path.join(base_path, "grnTruth.dat")) as f:
        FileRead = f.readlines()

    gt_dict = {line.split()[0]: line.split()[1:] for line in FileRead}

    # AMT greedy:
    with open(os.path.join(base_path, "AMT", "Results", "greedy.txt")) as f:
        FileRead = f.readlines()
    prediction_dict = {line.split()[0]: line.split()[1:] for line in FileRead}
    errors, running_length = evaluate_dictionaries(gt_dict, prediction_dict)
    fresults.write(
        "AMT-Greedy\tErrors:{}, running:{}, SER:{:.2f}%".format(
            errors, running_length, 100 * errors / running_length
        )
        + "\n"
    )

    # AMT kaldi:
    with open(os.path.join(base_path, "AMT", "Results", "Predictions.txt")) as f:
        FileRead = f.readlines()
    prediction_dict = {line.split()[0]: line.split()[1:] for line in FileRead}
    errors, running_length = evaluate_dictionaries(gt_dict, prediction_dict)
    fresults.write(
        "AMT-Kaldi\tErrors:{}, running:{}, SER:{:.2f}%".format(
            errors, running_length, 100 * errors / running_length
        )
        + "\n"
    )

    # OMR greedy:
    with open(os.path.join(base_path, "OMR", "Results", "greedy.txt")) as f:
        FileRead = f.readlines()
    prediction_dict = {line.split()[0]: line.split()[1:] for line in FileRead}
    errors, running_length = evaluate_dictionaries(gt_dict, prediction_dict)
    fresults.write(
        "OMR-Greedy\tErrors:{}, running:{}, SER:{:.2f}%".format(
            errors, running_length, 100 * errors / running_length
        )
        + "\n"
    )

    # OMR kaldi:
    with open(os.path.join(base_path, "OMR", "Results", "Predictions.txt")) as f:
        FileRead = f.readlines()
    prediction_dict = {line.split()[0]: line.split()[1:] for line in FileRead}
    errors, running_length = evaluate_dictionaries(gt_dict, prediction_dict)
    fresults.write(
        "OMR-Kaldi\tErrors:{}, running:{}, SER:{:.2f}%".format(
            errors, running_length, 100 * errors / running_length
        )
        + "\n"
    )

    # Light AMT-OMR:
    with open(
        os.path.join(base_path, "Light_AMT_OMR", "Results", "Predictions.txt")
    ) as f:
        FileRead = f.readlines()
    prediction_dict = {line.split()[0]: line.split()[1:] for line in FileRead}
    errors, running_length = evaluate_dictionaries(gt_dict, prediction_dict)
    fresults.write(
        "Light_AMT_OMR\tErrors:{}, running:{}, SER:{:.2f}%".format(
            errors, running_length, 100 * errors / running_length
        )
        + "\n"
    )

    # Light OMR-AMT:
    with open(
        os.path.join(base_path, "Light_OMR_AMT", "Results", "Predictions.txt")
    ) as f:
        FileRead = f.readlines()
    prediction_dict = {line.split()[0]: line.split()[1:] for line in FileRead}
    errors, running_length = evaluate_dictionaries(gt_dict, prediction_dict)
    fresults.write(
        "Light_OMR_AMT\tErrors:{}, running:{}, SER:{:.2f}%".format(
            errors, running_length, 100 * errors / running_length
        )
        + "\n"
    )

    list_WG_combinations = [
        "OMR1_AMT9",
        "OMR2_AMT8",
        "OMR3_AMT7",
        "OMR4_AMT6",
        "OMR5_AMT5",
        "OMR6_AMT4",
        "OMR7_AMT3",
        "OMR8_AMT2",
        "OMR9_AMT1",
    ]

    # WG combinations:
    for single_WG_combination in list_WG_combinations:
        with open(
            os.path.join(base_path, single_WG_combination, "Results", "Predictions.txt")
        ) as f:
            FileRead = f.readlines()
        prediction_dict = {line.split()[0]: line.split()[1:] for line in FileRead}
        errors, running_length = evaluate_dictionaries(gt_dict, prediction_dict)
        fresults.write(
            single_WG_combination
            + "\tErrors:{}, running:{}, SER:{:.2f}%".format(
                errors, running_length, 100 * errors / running_length
            )
            + "\n"
        )

    fresults.close()

    return


if __name__ == "__main__":
    src_OMR = "Results/OMR/"
    src_AMT = "Results/AMT"
    WG_dst = "Results"

    # Copy words path:
    words_src_path = os.path.join(
        os.path.split(os.path.split(src_OMR)[0])[0],
        "data",
        "train",
        "lang",
        "words.txt",
    )
    words_path = os.path.join(WG_dst, "words.txt")
    shutil.copy(words_src_path, words_path)

    if not os.path.exists(WG_dst):
        os.makedirs(WG_dst)

    # Copy original graphs:
    ExtremeCases(src_AMT, src_OMR, WG_dst, words_path)

    # COPY CONFUSION NETWORKS
    src_OMR = src_OMR.replace("_lattices", "_ConfNetworks")
    src_AMT = src_AMT.replace("_lattices", "_ConfNetworks")
    ConfusionNetworks(src_OMR, src_AMT, WG_dst)

    # Smith-Waterman approach:
    src_OMR = os.path.join(WG_dst, "OMR", "WG")
    src_AMT = os.path.join(WG_dst, "AMT", "WG")
    ObtainSWProbs(src_AMT, src_OMR, words_path)

    # Combining word graphs
    CombineWordGraphs(src_OMR, src_AMT, WG_dst, words_path)

    # Light WG combination (OMR -> AMT):
    CombineLightWordGraphs(src_OMR, src_AMT, "Light_OMR_AMT", words_path)

    # Light WG combination (AMT -> OMR):
    CombineLightWordGraphs(src_AMT, src_OMR, "Light_AMT_OMR", words_path)

    # Summary of results:
    results_summary(WG_dst)

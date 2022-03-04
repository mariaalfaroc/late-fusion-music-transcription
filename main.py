# -*- coding: utf-8 -*-

import os, shutil

import tensorflow as tf

import config
from experimentation import k_fold_experiment, k_fold_test_experiment

from word_graphs.smith_waterman import k_fold_multimodal_experiment as sw_k_fold_multimodal_experiment
from confusion_networks.cn_combination import k_fold_multimodal_experiment as cn_k_fold_multimodal_experiment
from word_graphs.wg_decoded_evaluation import k_fold_multimodal_experiment as cwg_k_fold_multimodal_experiment, k_fold_light_multimodal_experiment as light_cwg_k_fold_multimodal_experiment, k_fold_baseline_experiment as wg_k_fold_baseline_experiment
from scenarios.folds_creation import create_folds, create_folds_according_ser

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.config.list_physical_devices("GPU")

if __name__ == "__main__":
    epochs = 150
    scenarios = {"1": 2.5, "2": 2.2, "3": 2.0, "4": 4.0, "5": 3.7, "6": 3.0, "7": 10.5, "8": 9.0, "9": 6.3}

    # Evaluate AMT on Scenario X to be able to create test partitions based on model performance
    config.set_scenario(value="X")
    # AMT
    config.set_task(value="amt")
    config.set_data_globals()
    config.set_arch_globals(batch=4)
    print(f"Task == {config.task}")
    print(f"Scenario == {config.scenario}")
    k_fold_experiment(epochs)
    # Scenarios 1, 4, and 7 are the same as X for AMT
    shutil.copytree(src=str(config.output_dir / "amt"), dst=str(config.base_dir / "Experiments" / "Scenario1" / "amt"))
    shutil.copytree(src=str(config.output_dir / "amt"), dst=str(config.base_dir / "Experiments" / "Scenario4" / "amt"))
    shutil.copytree(src=str(config.output_dir / "amt"), dst=str(config.base_dir / "Experiments" / "Scenario7" / "amt"))
    # The rest of scenarios for AMT use the same model and the same vocabulary as in ScenarioX
    # Right now, we only need to copy those files to Scenarios 2 and 3
    for s in ["2", "3"]:
        os.makedirs(config.base_dir / "Experiments" / f"Scenario{s}", exist_ok=True)
        os.makedirs(config.base_dir / "Experiments" / f"Scenario{s}" / "amt", exist_ok=True)
        for f in os.listdir(config.output_dir / "amt"):
            os.makedirs(config.base_dir / "Experiments" / f"Scenario{s}" / "amt" / f, exist_ok=True)
            shutil.copyfile(str(config.output_dir / "amt" / f / "best_model.keras"), config.base_dir / "Experiments" / f"Scenario{s}" / "amt" / f / "best_model.keras")
            shutil.copyfile(str(config.output_dir / "amt" / f / "w2i.json"), config.base_dir / "Experiments" / f"Scenario{s}" / "amt" / f / "w2i.json")

    # Create folds for the rest of the scenarios to evaluate OMR
    for s, p_size in scenarios.items():
        if s in ["1", "4", "7"]:
            create_folds(p_size=p_size, scenario=s)
        elif s in ["2", "5", "8"]:
            create_folds_according_ser(p_size=p_size, scenario=s, symer_threshold=30)
        else:
            create_folds_according_ser(p_size=p_size, scenario=s, symer_threshold=10)

    # STAND-ALONE EVALUATION
    for s in scenarios.keys():
        config.set_scenario(value=s)
        if s in ["2", "3"]:
            config.set_task(value="amt")
            config.set_data_globals()
            config.set_arch_globals(batch=4)
            print(f"Task == {config.task}")
            print(f"Scenario == {config.scenario}")
            k_fold_test_experiment()
            if s == "2":
                shutil.copytree(src=str(config.output_dir / "amt"), dst=str(config.base_dir / "Experiments" / "Scenario5" / "amt"))
                shutil.copytree(src=str(config.output_dir / "amt"), dst=str(config.base_dir / "Experiments" / "Scenario8" / "amt"))
            else:
                shutil.copytree(src=str(config.output_dir / "amt"), dst=str(config.base_dir / "Experiments" / "Scenario6" / "amt"))
                shutil.copytree(src=str(config.output_dir / "amt"), dst=str(config.base_dir / "Experiments" / "Scenario9" / "amt"))
        # OMR
        config.set_task(value="omr")
        config.set_data_globals()
        config.set_arch_globals(batch=16)
        print(f"Task == {config.task}")
        print(f"Scenario == {config.scenario}")
        k_fold_experiment(epochs)

    # TODO: JJ code should be placed here

    # MULTIMODAL EVALUATION
    match = [2, 10, 20, 5]
    mismatch = [-1, 5, 10, 2,]
    gap_penalty = [-1, -2, -4, -1]
    for s in scenarios.keys():
        config.set_scenario(value=s)
        print(f"Scenario{config.scenario}")
        # BASELINE
        wg_k_fold_baseline_experiment()
        # 1) SMITH - WATERMAN
        sw_k_fold_multimodal_experiment(match=match, mismatch=mismatch, gap_penalty=gap_penalty)
        # 2) CONFUSION NETWORKS
        cn_k_fold_multimodal_experiment()
        # 3) COMBINED WORD GRAPHS
        cwg_k_fold_multimodal_experiment()
        # 4) LIGHT COMBINED WORD GRAPHS
        light_cwg_k_fold_multimodal_experiment()

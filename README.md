<p align="center">
  <a href=""><img src="https://i.imgur.com/Iu7CvC1.png" alt="PRAIG-logo" width="100"></a>
</p>

<h1 align="center">Late multimodal fusion for image and audio music transcription</h1>

<h4 align="center">Full text available <a href="https://www.sciencedirect.com/science/article/pii/S0957417422025106" target="_blank">here</a>.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9.0-orange" alt="Python">
  <img src="https://img.shields.io/badge/Tensorflow-%FFFFFF.svg?style=flat&logo=Tensorflow&logoColor=orange&color=white" alt="Tensorflow">
  <img src="https://img.shields.io/static/v1?label=License&message=MIT&color=blue" alt="License">
</p>


<p align="center">
  <a href="#about">About</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citations">Citations</a> •
  <a href="#acknowledgments">Acknowledgments</a> •
  <a href="#license">License</a>
</p>


## About

**We study four multimodal late-fusion approaches in order to merge the hypotheses of end-to-end Optical Music Recognition (OMR) and Automatic Music Transcription (AMT)[^1] systems in a word graph-based search space:** (i) one that addresses the fusion carried out by employing the Minimum Bayes Risk criterion,[^2] (ii) a second case that addresses the task from a lightly-supervised learning perspective,[^3] (iii) a third approach that follows a global alignment strategy,[^4] and (iv) a last procedure based on local alignment.[^5]

[^1]: It is important to clarify that the model we are referring to is actually an Audio-to-Score (A2S) model. At the time of conducting this research, we used the term AMT because the distinction between AMT and A2S did not exist in the literature. However, nowadays, there is a clear distinction between the two. AMT typically focuses on note-level transcription, encoding the acoustic piece in terms of onset, offset, pitch values, and the musical instrument of the estimated notes. In contrast, A2S aims to achieve a score-level codification.

[^2] Xu, H., Povey, D., Mangu, L., & Zhu, J. (2011). Minimum Bayes Risk Decoding and System Combination Based on a Recursion for Edit Distance. Comput. Speech Lang., 25, 802–828.
[^3] Fainberg, J., Klejch, O., Renals, S., & Bell, P. (2019). Lattice-Based Lightly-Supervised Acoustic Model Training. In Interspeech 20th Annual Conference of the International Speech Communication Association (pp. 1596–1600). Graz, Austria.
[^4] Granell, E., & Mart ́ınez-Hinarejos, C.-D. (2015). Combining handwriting and speech recognition for transcribing historical handwritten documents. In 13th International Conference on Document Analysis and Recognition (ICDAR) (pp. 126–130).
[^5] de la Fuente, C., Valero-Mas, J. J., Castellanos, F. J., & Calvo-Zaragoza, J. (2021). Multimodal image and audio music transcription. Int. J. Multimed. Inf. Retr., (pp. 1–8).

<p align="center">
  <img src="scheme.jpg" alt="content" style="border: 1px solid black; width: 800px;">
</p>



## How To Use

### Dataset

We use the [**Camera-PrIMuS**](https://grfia.dlsi.ua.es/primus/) dataset.

The Camera-PrIMuS dataset contains 87&nbsp;678[^6] real-music incipits[^7], each represented by six files: (i) the Plaine and Easie code source, (ii) an image with the rendered score, (iii) a distorted image, (iv) the musical symbolic representation of the incipit both in Music Encoding Initiative format (MEI) and (v) in an on-purpose simplified encoding (semantic encoding), and (vi) a sequence containing the graphical symbols shown in the score with their position in the staff without any musical meaning (agnostic encoding).

[^6]: In this work, we consider 22&nbsp;285 samples out of the total 87&nbsp;678 that constitute the complete Camera-PrIMuS dataset. This selection resulted from a data curation process, primarily involving the removal of samples containing long multi-rests. These music events contribute minimally to the length of the score image but may span a large number of frames in the audio signal.

[^7]: An incipit is a sequence of notes, typically the first ones, used for identifying a melody or musical work.

To obtain the corresponding audio files, we must convert one of the provided representations to MIDI and then synthesize the MIDI data. We have opted to convert the semantic representation, as there is a publicly available semantic-to-MIDI converter. Once we have obtained the MIDI files, we render them using FluidSynth.

The specific steps to follow are:
1) Download the semantic-to-MIDI converter from [here](https://grfia.dlsi.ua.es/primus/primus_converter.tgz) and **place the `omr-3.0-SNAPSHOT.jar` file in the [`dataset`](dataset) folder**.
2) Download a [General MIDI SounFont (sf2)](https://sites.google.com/site/soundfonts4u/#h.p_biJ8J359lC5W). We recommend downloading the [SGM-v2.01 soundfont](https://drive.google.com/file/d/12zSPpFucZXFg-svKeu6dm7-Fe5m20xgJ/view) as this code has been tested using this soundfont. **Place the sf2 file in the [`dataset`](dataset) folder.**

## Experiments

We want to know how the differences between the performances of OMR and AMT models affect the combined prediction. For that, we consider three levels of unimodal model performance:
1) High (SER of around 30%) 
2) Medium (SER of around 20%)
3) Low (SER of around 10%)

We then evaluate all possible combinations of those three levels with OMR and AMT models. This accounts for a total of 9 evaluation scenarios.

To replicate our experiments, you will first need to meet certain requirements specified in the [`Dockerfile`](Dockerfile). Alternatively, you can set up a virtual environment if preferred. Once you have prepared your environment (either a Docker container or a virtual environment) and followed the steps in the [dataset](#dataset) section, you are ready to begin. Follow this recipe to replicate our experiments:

> **Important note: To execute the following code, both Java and FluidSynth must be installed.**

```bash
$ cd dataset
$ sh prepare_data.sh
$ cd ..
$ python main.py
```

## Citations

```bibtex
@article{alfaro2023late,
  title        = {{Late multimodal fusion for image and audio music transcription}},
  author       = {Alfaro-Contreras, Mar{\'i}a and Valero-Mas, Jose J. and I{\~n}esta, Jose M. and Calvo-Zaragoza, Jorge},
  journal      = {{Expert Systems with Applications}},
  volume       = {216},
  pages        = {119491--119500},
  year         = {2023},
  publisher    = {Elsevier},
  doi          = {10.1016/j.eswa.2022.119491},
}
```

## Acknowledgments

This work is part of the I+D+i PID2020-118447RA-I00 ([MultiScore](https://sites.google.com/view/multiscore-project)) project, funded by MCIN/AEI/10.13039/501100011033.

## License
This work is under a [MIT](LICENSE) license.

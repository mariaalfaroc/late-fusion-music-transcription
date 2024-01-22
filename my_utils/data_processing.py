import joblib
from typing import List, Dict, Tuple, Union

import cv2
import librosa
import numpy as np
from sklearn.utils import shuffle

from networks.models import INPUT_HEIGHT, POOLING_FACTORS

MEMORY = joblib.memory.Memory("./joblib_cache", mmap_mode="r", verbose=0)


################################################################# INPUT DATA PREPROCESSING:


def resize_image(image: np.ndarray, new_height: int) -> np.ndarray:
    new_width = int(new_height * image.shape[1] / image.shape[0])
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


# Preprocess image:
# 1) Read from path
# 2) Convert to grayscale
# 3) Normalize
# 4) Resize preserving aspect ratio
@MEMORY.cache
def preprocess_image(image_path: str) -> Tuple[np.ndarray, int]:
    img = cv2.imread(image_path, 0)  # Read as grayscale
    img = (255.0 - img) / 255.0
    img = resize_image(img, INPUT_HEIGHT["omr"])
    img = np.expand_dims(img, -1)
    img = img.astype("float32")
    return img, img.shape[1] // POOLING_FACTORS["omr"]["width_reduction"]


# Preprocess audio:
# 1) Read from path
# 2) Obtain CQT spectrogram
# 3) Normalize
# 4) Resize preserving aspect ratio
@MEMORY.cache
def preprocess_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(audio_path, sr=22050)
    spec = librosa.cqt(
        audio,
        sr=sr,
        hop_length=512,
        n_bins=120,
        bins_per_octave=24,
    )
    spec = librosa.amplitude_to_db(abs(spec), ref=np.max)
    spec = np.flip(spec, 0)
    spec = (80.0 + spec) / 80.0
    spec = resize_image(spec, INPUT_HEIGHT["amt"])
    spec = np.expand_dims(spec, -1)
    spec = spec.astype("float32")
    return spec, spec.shape[1] // POOLING_FACTORS["amt"]["width_reduction"]


def preprocess_input(task: str, input_path: str):
    if task == "omr":
        return preprocess_image(input_path)
    elif task == "amt":
        return preprocess_audio(input_path)
    else:
        raise ValueError("Invalid task")


# Preprocess label:
# 1) Read from path
# 2) Split by encoding grammar
# 3) Convert to integer if training
def preprocess_label(
    label_path: str, training: bool, w2i: Dict[str, int]
) -> Union[Tuple[List[int], int], List[str]]:
    label = open(label_path).read().split()
    if training:
        label = [w2i[w] for w in label]
        return label, len(label)
    return label


################################################################# CTC PREPROCESSING:


# CTC-loss preprocess function -> (xi, yi)
# xi[0] -> images, zero-padded to the maximum image width found
# xi[1] -> real width (after the CNN) of the images
# xi[2] -> labels, CTC-blank-padded to the maximum label length found
# xi[3] -> real length of the labels
# yi[0] -> dummy value for CTC-loss
def ctc_preprocess(
    images: list,
    labels: list,
    blank_index: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    # Unzip variables
    images, images_len = list(zip(*images))
    labels, labels_len = list(zip(*labels))
    # Obtain the current batch size
    num_samples = len(images)
    # Zero-pad images to maximum batch image width
    max_width = max(images, key=np.shape).shape[1]
    images = np.array(
        [
            np.pad(i, pad_width=((0, 0), (0, max_width - i.shape[1]), (0, 0)))
            for i in images
        ],
        dtype="float32",
    )
    images_len = np.array(images_len, dtype="int32").reshape((num_samples, 1))
    # CTC-blank-pad labels to maximum batch label length
    max_length = len(max(labels, key=len))
    labels = np.array(
        [i + [blank_index] * (max_length - len(i)) for i in labels], dtype="int32"
    )
    labels_len = np.array(labels_len, dtype="int32").reshape((num_samples, 1))
    # Format data
    xi = {
        "image": images,
        "image_len": images_len,
        "label": labels,
        "label_len": labels_len,
    }
    yi = {"ctc_loss": np.zeros(shape=(num_samples, 1), dtype="float32")}
    return xi, yi


# Train data generator
def train_data_generator(
    task: str,
    images_files: List[str],
    labels_files: List[str],
    w2i: Dict[str, int],
    batch_size: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    images_files, labels_files = shuffle(images_files, labels_files, random_state=42)

    size = len(images_files)
    start = 0
    while True:
        end = min(start + batch_size, size)
        images = [
            preprocess_input(task=task, input_path=i) for i in images_files[start:end]
        ]
        labels = [
            preprocess_label(label_path=i, training=True, w2i=w2i)
            for i in labels_files[start:end]
        ]
        xi, yi = ctc_preprocess(images, labels, blank_index=len(w2i))
        yield xi, yi

        if end == size:
            start = 0
            # Due to the current training set up (model.fit() is called one epoch at a time),
            # it does not make sense to shuffle the data in this step
            # as the generator is stopped after all the training data is seen
            # This is why it is important to shuffle the data at the very beginning,
            # so that at each epoch it is seen in a different order
            # Uncomment the following line, if we were to call model.fit() for longer that one epoch
            # images_files, labels_files = shuffle(
            #     images_files, labels_files, random_state=42
            # )
        else:
            start = end

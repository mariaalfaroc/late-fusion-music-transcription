import os
import requests
import tarfile
import shutil

import fire
from music21 import converter
from midi2audio import FluidSynth


SOUND_FONT = "SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2"


def download_and_extract_camera_primus_dataset():
    file_path = "CameraPrIMuS.tgz"
    extract_path = "."

    # Download dataset
    response = requests.get(
        url="https://grfia.dlsi.ua.es/primus/packages/CameraPrIMuS.tgz"
    )
    with open(file_path, "wb") as file:
        file.write(response.content)
    # Extract dataset
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(extract_path)
    # Remove tar file
    os.remove(file_path)


def clean_dataset():
    with open("clean_samples.dat", "r") as f:
        clean_files = f.read().splitlines()

    for f in os.listdir("Corpus"):
        # Check f is a folder
        if os.path.isdir(os.path.join("Corpus", f)):
            # Check if f is in the clean files list
            if f not in clean_files:
                # Remove folder
                shutil.rmtree(os.path.join("Corpus", f))


def mei_to_midi(mei_file_path, midi_file_path):
    # Load the MEI file
    score = converter.parse(mei_file_path)

    # Export to MIDI
    _ = score.write("midi", fp=midi_file_path)


def create_multimodal_dataset():
    # Copy problematic sample
    shutil.copy(
        "110002867-1_1_1.wav",
        os.path.join("Corpus", "110002867-1_1_1", "110002867-1_1_1.wav"),
    )
    # Get all MEI files
    mei_files = []
    for foldername, subfolders, filenames in os.walk("Corpus"):
        for filename in filenames:
            if filename.endswith(".mei"):
                mei_files.append(os.path.join(foldername, filename))

    # Convert MEI to WAV
    fs = FluidSynth(sample_rate=22050, sound_font=SOUND_FONT)
    for mei_file in mei_files:
        # Convert MEI to MIDI
        midi_file = mei_file.replace(".mei", ".mid")
        # Catch semantic conversor errors
        if not os.path.exists(midi_file):
            mei_to_midi(mei_file, midi_file)
        # Convert MIDI to WAV
        wav_file = midi_file.replace(".mid", ".wav")
        # Catch problematic files
        if not os.path.exists(wav_file):
            fs.midi_to_audio(midi_file, wav_file)


if __name__ == "__main__":
    fire.Fire(
        {
            "download_and_extract_camera_primus_dataset": download_and_extract_camera_primus_dataset,
            "clean_dataset": clean_dataset,
            "create_multimodal_dataset": create_multimodal_dataset,
        }
    )

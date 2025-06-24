# SoloBass: An AI Bass Player

SoloBass is a project that uses a chord-conditioned LSTM (Long Short-Term Memory) network to generate musical basslines. It can be used for both real-time performance and offline processing of MIDI files.

This project analyzes the chords from a guitar track and generates a corresponding bass part note by note.

## Features

  * **Real-time Generation**: Listens to a live MIDI input for chords and generates a bassline in real-time using a synthesizer.
  * **Offline Generation**: Takes a MIDI file containing guitar chords and generates a new MIDI file with an added bass track.
  * **Data Augmentation**: The training process automatically transposes the source MIDI files to create a more robust dataset.
  * **Configurable**: All major hyperparameters, file paths, and model architecture details are centralized in `config.py` for easy modification.

## Requirements

  * **Python 3.11**
  * A SoundFont file (`.sf2`) for bass sounds (for real-time mode). A default `bass.sf2` is suggested.
  * The python packages listed in `requirements.txt`.

## Setup

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/hitoshitoshi/SoloBass.git
    cd SoloBass
    ```

2.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare MIDI data**

      * Create a folder named `midi_files` (or as configured in `config.py`).
      * Place MIDI files containing guitar and bass tracks into this folder for training.

4.  **Get a SoundFont**

      * For the real-time player (`SoloBass.py`), you will need a SoundFont file (e.g., `bass.sf2`). Download one and place it in the root directory or provide the path via the command line.

## How to Use

### Step 1: Train the Model

First, you must train the LSTM model on your MIDI dataset.

```bash
python train.py
```

  * This script will process the files in the `midi_files` directory, create a cached dataset (`cached_dataset.npz`), train the model, and save the learned weights to `./saved_models/unrolled_lstm.weights.h5`.
  * If you want to retrain the model even if a weights file already exists, use the `--force-train` flag:
    ```bash
    python train.py --force-train
    ```

### Step 2 (Option A): Generate a Bassline for a MIDI file

Use `testModel.py` to add a generated bassline to an existing MIDI file.

**Usage:**

```bash
python testModel.py <input_midi_path> <output_midi_path> [--temperature <value>]
```

  * **`input_midi_path`**: Path to the source MIDI file with guitar chords.
  * **`output_midi_path`**: Path where the new MIDI file with the generated bassline will be saved.
  * **`--temperature`** (optional): A float value (e.g., 1.0) that controls the randomness of note selection. Higher values lead to more variation. Defaults to 1.0.

### Step 2 (Option B): Use the Real-Time Bass Player

Use `SoloBass.py` to launch the real-time AI bass player that listens to your MIDI controller.

1.  **Find your MIDI port:**
    Run the script without arguments to list available MIDI input devices and their port numbers.

    ```bash
    python SoloBass.py
    ```

2.  **Run the real-time player:**

    ```bash
    python SoloBass.py --midi_port <port_number> --soundfont <path_to_sf2> [--temperature <value>]
    ```

      * **`--midi_port`**: The integer index of your MIDI device from the list generated in the previous step.
      * **`--soundfont`**: Path to your bass SoundFont file (e.g., `bass.sf2`).
      * **`--temperature`** (optional): Controls the randomness of note generation. Defaults to 1.0.

    Now, play chords on your MIDI controller, and SoloBass will generate and play a bassline in response\!

## File Descriptions

  * **`train.py`**: Script to train the neural network on the dataset in `midi_files`.
  * **`testModel.py`**: Generates a bassline for a given input MIDI file and saves the output.
  * **`SoloBass.py`**: The main real-time application that listens for MIDI input and plays a generated bassline.
  * **`data_preparation.py`**: Contains all functions for parsing, quantizing, and preparing MIDI data for training.
  * **`models.py`**: Defines the Keras/TensorFlow LSTM model architectures, for both training and single-step generation.
  * **`config.py`**: A centralized file for all hyperparameters, file paths, and constants.
  * **`requirements.txt`**: A list of all the python dependencies for this project.

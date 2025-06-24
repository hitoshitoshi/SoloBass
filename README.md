# SoloBass: AI Bass Player

SoloBass is an AI-powered bass player that listens to guitar chords and generates a matching bassline in real time. It uses a recurrent neural network (LSTM) to learn the relationship between chord progressions and bass melodies from a collection of MIDI files.

## Features

  * **Real-time Performance**: Connect a MIDI keyboard or controller to play chords, and SoloBass will generate and play a bassline live.
  * **Offline Generation**: Add a generated bassline to an existing MIDI file containing guitar chords.
  * **Customizable Training**: Train the model on your own collection of MIDI files to create a bass player with a unique style.

## How It Works

The project uses a chord-conditioned Long Short-Term Memory (LSTM) network built with TensorFlow/Keras.

1.  **Data Preparation**: The `data_preparation.py` script processes a folder of MIDI files. It extracts guitar and bass tracks, quantizes them into a time-series format, and slices them into windows suitable for training. Guitar parts are converted into a multi-hot vector representing the active notes (chords) at each timestep, while the bass part is converted into a sequence of note tokens.
2.  **Modeling**: The `models.py` script defines two model architectures:
      * An **unrolled model** for efficient, stateless training on sequences of data.
      * A **single-step model** with a stateful LSTM for real-time, step-by-step generation.
3.  **Training**: The `train.py` script orchestrates the training process. It builds a dataset from your MIDI files (or loads a cached one), trains the unrolled model, and saves the learned weights.
4.  **Inference**:
      * For offline testing, `testModel.py` loads the trained weights, processes an input MIDI file's chords, and generates a full bassline, saving the result to a new MIDI file.
      * For live performance, `SoloBass.py` loads the weights into the stateful model. It listens for MIDI input to update its internal chord representation and uses the model to predict the next bass note, which is then played via FluidSynth.

## Setup & Installation

**1. Clone the Repository**

```bash
git clone <repository-url>
cd SoloBass
```

**2. Install Python Dependencies**

Create a virtual environment and install the required packages from `requirements.txt`.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

The requirements include `tensorflow`, `numpy`, `pretty_midi`, `mido`, `pyfluidsynth`, and `tqdm`.

**3. Install FluidSynth**

The `pyfluidsynth` library requires the FluidSynth software to be installed on your system.

  * **macOS (via Homebrew):** `brew install fluidsynth`
  * **Debian/Ubuntu:** `sudo apt-get install fluidsynth`
  * **Windows:** Download and install from the official [FluidSynth website](https://www.google.com/search?q=https://www.fluidsynth.org/get-started/).

**4. Download a Soundfont**

You need a SoundFont (`.sf2`) file for audio output. A bass guitar soundfont is recommended. Place it in the project's root directory or provide the path via the command line.

## Usage

### 1\. Data Preparation

Place your training MIDI files (containing both guitar/chord tracks and bass tracks) into the `midi_files/` directory. The scripts identify instruments based on the General MIDI program numbers defined in `data_preparation.py`.

### 2\. Training the Model

Run the training script. This will process the MIDI files, create a `cached_dataset.npz`, and save the trained model weights to `saved_models/unrolled_lstm.weights.h5`.

```bash
python train.py
```

  * Use the `--force-train` flag to retrain the model even if a weights file already exists.

### 3\. Offline Generation (Create a Bassline for a MIDI file)

Use the `testModel.py` script to add a bassline to an existing MIDI file.

```bash
python testModel.py <path/to/input_midi.mid> <path/to/output_midi.mid>
```

  * `--temperature <value>`: Adjust the randomness of the note selection. A higher value (e.g., 1.2) leads to more variation, while a lower value (e.g., 0.8) sticks closer to the model's prediction.

### 4\. Real-Time Performance

Use `SoloBass.py` to start the live AI bass player.

**First, find your MIDI input port:**

```bash
python SoloBass.py
```

This will list the available MIDI ports and their index numbers.

**Then, run the script with your chosen port:**

```bash
python SoloBass.py --midi_port <port_index> --soundfont <path/to/bass.sf2>
```

  * `--midi_port <index>`: The index of your MIDI controller.
  * `--soundfont <path>`: Path to your `.sf2` file (defaults to `bass.sf2`).
  * `--temperature <value>`: Adjust the creativity of the live performance.

Now, play chords on your MIDI controller, and SoloBass will accompany you on the bass\!

## File Descriptions

  * `SoloBass.py`: Main script for real-time, interactive bassline generation.
  * `train.py`: Handles the dataset preparation and model training loop.
  * `testModel.py`: Offline script to generate a bassline for an entire MIDI file.
  * `models.py`: Contains the Keras/TensorFlow definitions for the LSTM models (both unrolled for training and stateful for real-time use).
  * `data_preparation.py`: A library of functions for parsing MIDI files, extracting instrument tracks, and quantizing note data into arrays.
  * `config.py`: A central file for all hyperparameters and settings, such as MIDI pitch ranges, model dimensions, learning rate, and sequence length.
  * `requirements.txt`: A list of the Python packages required for this project.
  * `.gitignore`: Specifies files and directories to be ignored by Git, such as the dataset cache and virtual environments.

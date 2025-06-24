# SoloBass: A Real-time AI Bass Player

SoloBass is a deep learning model that generates a bass line in real-time to accompany a guitarist. It can also be used to add a bass line to a MIDI file that contains guitar chords.

## Features

  - **Real-time performance:** Listens to a MIDI guitar input and generates a bass line on the fly.
  - **Offline generation:** Adds a bass line to a MIDI file containing guitar chords.
  - **Customizable:** The model's architecture and training parameters can be configured.

## How it Works

SoloBass uses a recurrent neural network (RNN) with LSTM cells to predict a bass note based on the current guitar chord and the previous bass note.

The model is trained on a dataset of MIDI files containing both guitar and bass tracks. The `data_preparation.py` script extracts the guitar chords and bass lines from the MIDI files, quantizes them, and slices them into sequences for training.

The `train.py` script builds and trains the model using the prepared dataset. The trained model weights are saved to a file.

The `testModel.py` script loads the trained model and uses it to add a bass line to a given MIDI file.

The `SoloBass.py` script runs the model in real-time, taking MIDI input from a guitar and sending the generated bass line to a software synthesizer.

## Requirements

The following dependencies are required to run SoloBass:

  - tensorflow
  - numpy
  - pretty\_midi
  - mido
  - pyfluidsynth
  - tqdm

You can install them using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1.  Place your MIDI files in the `midi_files` directory.
2.  Run the `train.py` script to train the model:

<!-- end list -->

```bash
python train.py
```

The script will cache the prepared dataset to `cached_dataset.npz` and save the trained model weights to `saved_models/unrolled_lstm.weights.h5`.

### Generating a Bass Line for a MIDI File

To add a bass line to a MIDI file, use the `testModel.py` script:

```bash
python testModel.py <input_midi> <output_midi>
```

  - `<input_midi>`: The path to the input MIDI file with guitar chords.
  - `<output_midi>`: The path to write the output MIDI file.

You can also adjust the sampling temperature for note generation:

```bash
python testModel.py <input_midi> <output_midi> --temperature 1.2
```

### Real-time Performance

To use SoloBass in real-time, you will need a MIDI keyboard or a MIDI guitar controller, and a soundfont file for the bass sound.

1.  Connect your MIDI controller to your computer.
2.  Run the following command to list the available MIDI ports:

<!-- end list -->

```bash
python SoloBass.py
```

3.  Run the script again, specifying the MIDI port to use and the path to your soundfont file:

<!-- end list -->

```bash
python SoloBass.py --midi_port <port_number> --soundfont <soundfont_file>
```

  - `<port_number>`: The index of the MIDI input port to use.
  - `<soundfont_file>`: The path to the soundfont file (.sf2).

You can also adjust the sampling temperature:

```bash
python SoloBass.py --midi_port <port_number> --soundfont <soundfont_file> --temperature 0.8
```

## Files

  - `SoloBass.py`: The main script for real-time bass generation.
  - `testModel.py`: A script to add a generated bass line to a MIDI file.
  - `train.py`: The script to train the model.
  - `data_preparation.py`: Contains functions for parsing and preparing MIDI data.
  - `models.py`: Defines the model architecture.
  - `config.py`: Contains global settings and hyperparameters.
  - `requirements.txt`: A list of the project's dependencies.
  - `.gitignore`: Specifies which files to ignore in a Git repository.

# add_generated_notes.py  ──  bass‑line generation demo
#
#   Given a MIDI file, read the guitar‑chord timeline, run the trained
#   chord‑conditioned LSTM, and write a new MIDI that contains
#       • guitar chords (original) and
#       • a freshly generated BASS line.
#
# Author: ChatGPT (o3)

import os
import numpy as np
import pretty_midi
import tensorflow as tf

import data_preparation
import models
from config import (                # keep one source of truth
    REST_TOKEN,
    GUITAR_LOWEST_PITCH,
    STEPS_PER_QUARTER,
    BASS_LOWEST_PITCH
)

WEIGHTS_PATH = "unrolled_lstm.weights.h5"

# ───────────────────────── SAMPLING ───────────────────────── #

def sample_note(prob_dist, temperature=1.0):
    """Temperature‑controlled sampling from a probability distribution."""
    logp = np.log(prob_dist + 1e-9) / temperature
    p    = np.exp(logp) / np.sum(np.exp(logp))
    return np.random.choice(len(prob_dist), p=p)

# ───────────────────────── MIDI HELPERS ───────────────────────── #

def chord_array_to_midi_instrument(
        chord_array,
        tempo_us_per_beat=500000,           # 120 BPM
        program=26                          # Jazz Guitar
):
    """
    Convert multi‑hot chord array → PrettyMIDI Instrument (guitar chords).
    Consecutive identical frames are merged into sustained notes.
    """
    inst = pretty_midi.Instrument(program=program, name="Guitar Chords")
    beat_dur = tempo_us_per_beat / 1_000_000.0
    step     = beat_dur / STEPS_PER_QUARTER

    active = tuple()          # current active idx tuple
    start  = 0.0

    for t, frame in enumerate(chord_array):
        idx_now = tuple(np.where(frame > 0.5)[0])
        if idx_now != active:           # chord change
            if active:                  # finish previous chord
                for idx in active:
                    pitch = idx + GUITAR_LOWEST_PITCH
                    inst.notes.append(
                        pretty_midi.Note(velocity=90,
                                         pitch=pitch,
                                         start=start,
                                         end=t*step)
                    )
            active, start = idx_now, t*step

    # close final chord
    end_time = len(chord_array) * step
    if active:
        for idx in active:
            pitch = idx + GUITAR_LOWEST_PITCH
            inst.notes.append(
                pretty_midi.Note(velocity=90,
                                 pitch=pitch,
                                 start=start,
                                 end=end_time)
            )
    return inst


def tokens_to_bass_instrument(tokens,
                              tempo_us_per_beat=500000,
                              program=32):          # Acoustic Bass
    """
    Merge consecutive identical bass tokens into sustained notes.
    """
    inst = pretty_midi.Instrument(program=program, name="Generated Bass")
    beat_dur = tempo_us_per_beat / 1_000_000.0
    step     = beat_dur / STEPS_PER_QUARTER

    t = 0
    while t < len(tokens):
        tok = tokens[t]
        if tok == REST_TOKEN:
            t += 1
            continue

        # count how long the same note lasts
        start = t
        while t < len(tokens) and tokens[t] == tok:
            t += 1
        end = t

        pitch = tok + BASS_LOWEST_PITCH
        inst.notes.append(
            pretty_midi.Note(velocity=100,
                             pitch=pitch,
                             start=start*step,
                             end=end*step)
        )
    return inst

# ───────────────────────── MAIN PIPELINE ───────────────────────── #

def add_generated_bass_to_midi(input_midi,
                               output_midi,
                               temperature=1.0):
    """
    • Extract guitar‑chord array from `input_midi`.
    • Run LSTM step‑by‑step to create a bass token for every frame.
    • Write new MIDI containing guitar chords + generated bass line.
    """
    if not os.path.exists(input_midi):
        raise FileNotFoundError(input_midi)

    # 1. Parse MIDI → (bass_array, chord_array).  We ignore bass_array here.
    bass_arr, chord_arr = data_preparation.parse_midi_file(input_midi)
    steps = len(chord_arr)
    if steps == 0:
        print("No chord data; nothing to do.")
        return

    # 2. Load trained weights into real‑time model
    unrolled = models.build_unrolled_model()
    unrolled.load_weights(WEIGHTS_PATH)

    rt = models.build_single_step_model()
    for layer in rt.layers:
        try:
            layer.set_weights(unrolled.get_layer(layer.name).get_weights())
        except ValueError:
            pass
    rt.get_layer("lstm").reset_states()

    # 3. Autoregressive generation
    generated = []
    prev_tok  = REST_TOKEN
    for t in range(steps):
        chord_vec = chord_arr[t].reshape(1, 1, -1).astype(np.float32)
        note_in   = np.array([[prev_tok]], dtype=np.int32)
        pred, _, _ = rt.predict([note_in, chord_vec], verbose=0)
        next_tok   = sample_note(pred[0], temperature)
        generated.append(next_tok)
        prev_tok = next_tok

    # 4. Build output MIDI
    tempo = 500000          # 120 BPM; adjust if desired
    pm = pretty_midi.PrettyMIDI()

    pm.instruments.append(
        chord_array_to_midi_instrument(chord_arr,
                                       tempo_us_per_beat=tempo,
                                       program=26)      # Jazz Guitar
    )
    pm.instruments.append(
        tokens_to_bass_instrument(generated,
                                  tempo_us_per_beat=tempo,
                                  program=33)           # Acoustic Bass
    )
    pm.write(output_midi)
    print("Wrote", output_midi)


# ───────────────────────── CLI EXAMPLE ───────────────────────── #

if __name__ == "__main__":
<<<<<<< HEAD
    add_generated_bass_to_midi("oasis.mid", "oasis_with_generated_bass.mid", temperature=1.0)
=======
    input_mid  = "TP.mid"            # Input MIDI file containing the original chord information.
    output_mid = "TP_with_generated.mid"  # Output file to be created.
    add_generated_notes_to_midi(input_mid, output_mid, temperature=1.0)
>>>>>>> c85172d11d4521fb01369a673d092f62abc7cc18

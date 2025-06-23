import os
import numpy as np
import pretty_midi
from tqdm import tqdm

from config import (
    MIDI_FOLDER,
    SEQUENCE_LENGTH,
    # pitch constants
    BASS_LOWEST_PITCH, BASS_HIGHEST_PITCH, BASS_REST_TOKEN,
    GUITAR_LOWEST_PITCH, GUITAR_HIGHEST_PITCH, CHORD_VECTOR_SIZE,
    STEPS_PER_QUARTER, DEFAULT_BPM
)

# ───────────────────────── INSTRUMENT IDS ───────────────────────── #

# GM program numbers (0‑based).  Also include +1 for files that are 1‑based.
GUITAR_PROGRAMS = {*(range(24, 32)), *(range(25, 33))}
BASS_PROGRAMS   = {*(range(32, 40)), *(range(33, 41))}

# ───────────────────────── TRACK EXTRACTION ───────────────────────── #

def extract_bass_and_guitar_tracks(midi_path):
    """Return two pretty_midi.Instrument objects: bass_inst, guitar_inst."""
    pm = pretty_midi.PrettyMIDI(midi_path)

    bass_inst   = pretty_midi.Instrument(program=32)  # placeholder programs
    guitar_inst = pretty_midi.Instrument(program=27)

    for inst in pm.instruments:
        prog = inst.program
        if prog in BASS_PROGRAMS:
            bass_inst.notes.extend(inst.notes)
        elif prog in GUITAR_PROGRAMS:
            guitar_inst.notes.extend(inst.notes)
    return bass_inst, guitar_inst

# ───────────────────────── QUANTISATION ───────────────────────── #

def quantize_bass_to_array(inst, steps_per_quarter=4, default_bpm=120):
    """Monophonic bass → 1‑D token array (REST_TOKEN when silent)."""
    if not inst.notes:
        return np.array([], dtype=np.int32)

    max_end = max(n.end for n in inst.notes)
    step_dur = (120 / default_bpm) / steps_per_quarter
    total_steps = int(np.ceil(max_end / step_dur))

    events = [(n.start, n.pitch, True) for n in inst.notes] + \
             [(n.end,   n.pitch, False) for n in inst.notes]
    events.sort(key=lambda x: x[0])

    out = np.full(total_steps, BASS_REST_TOKEN, dtype=np.int32)
    current = BASS_REST_TOKEN
    e_idx = 0

    for s in range(total_steps):
        t = s * step_dur
        while e_idx < len(events) and events[e_idx][0] <= t:
            _, p, on = events[e_idx]
            current = p if on else (BASS_REST_TOKEN if p == current else current)
            e_idx += 1

        if (current != BASS_REST_TOKEN and
            BASS_LOWEST_PITCH <= current <= BASS_HIGHEST_PITCH):
            out[s] = current - BASS_LOWEST_PITCH
    return out


def quantize_guitar_to_chord_array(inst, steps_per_quarter=4, default_bpm=120):
    """Polyphonic guitar → (time, CHORD_VECTOR_SIZE) multi‑hot array."""
    if not inst.notes:
        return np.zeros((0, CHORD_VECTOR_SIZE), dtype=np.float32)

    max_end = max(n.end for n in inst.notes)
    step_dur = (120 / default_bpm) / steps_per_quarter
    total_steps = int(np.ceil(max_end / step_dur))

    events = [(n.start, n.pitch, True) for n in inst.notes] + \
             [(n.end,   n.pitch, False) for n in inst.notes]
    events.sort(key=lambda x: x[0])

    out = np.zeros((total_steps, CHORD_VECTOR_SIZE), dtype=np.float32)
    active = set()
    e_idx = 0

    for s in range(total_steps):
        t = s * step_dur
        while e_idx < len(events) and events[e_idx][0] <= t:
            _, p, on = events[e_idx]
            (active.add if on else active.discard)(p)
            e_idx += 1

        for p in active:
            if GUITAR_LOWEST_PITCH <= p <= GUITAR_HIGHEST_PITCH:
                idx = p - GUITAR_LOWEST_PITCH
                out[s, idx] = 1.0
    return out

# ───────────────────────── PARSER ───────────────────────── #

def parse_midi_file(midi_path, transpose=0):
    """Return (bass_tokens, guitar_chords)."""
    bass, git = extract_bass_and_guitar_tracks(midi_path)

    if transpose:
        for inst in (bass, git):
            for n in inst.notes:
                n.pitch += transpose

    bass_arr   = quantize_bass_to_array(
        bass, steps_per_quarter=STEPS_PER_QUARTER, default_bpm=DEFAULT_BPM)
    chord_arr  = quantize_guitar_to_chord_array(
        git,  steps_per_quarter=STEPS_PER_QUARTER, default_bpm=DEFAULT_BPM)

    # pad to equal length
    L = max(len(bass_arr), len(chord_arr))
    if len(bass_arr)  < L:
        bass_arr  = np.pad(bass_arr,  (0, L-len(bass_arr)),  constant_values=BASS_REST_TOKEN)
    if len(chord_arr) < L:
        chord_arr = np.pad(chord_arr, ((0, L-len(chord_arr)), (0, 0)))

    return bass_arr, chord_arr

# ───────────────────────── WINDOW SLICING ───────────────────────── #

def slice_into_windows(bass_arr, chord_arr, T=SEQUENCE_LENGTH):
    """Return (X_notes, X_chords, y_notes) for one song."""
    L = len(bass_arr)
    if L < T:
        return None, None, None

    Xn, Xc, Yn = [], [], []
    for i in range(0, L - T, T):
        Xn.append(bass_arr[i       : i+T-1])
        Xc.append(chord_arr[i      : i+T-1, :])
        Yn.append(bass_arr[i+1     : i+T])
    if not Xn:
        return None, None, None

    return (np.asarray(Xn, dtype=np.int32),
            np.asarray(Xc, dtype=np.float32),
            np.asarray(Yn, dtype=np.int32))

# ───────────────────────── DATASET BUILDER ───────────────────────── #

def build_training_dataset(folder=MIDI_FOLDER,
                           T=SEQUENCE_LENGTH,
                           transpose_range=(-6, 7)):
    """Walk folder, augment by key‑shift, assemble full dataset."""
    Xn_all, Xc_all, Yn_all = [], [], []

    midi_files = [os.path.join(folder, f)
                  for f in os.listdir(folder)
                  if f.lower().endswith('.mid')]

    for path in tqdm(midi_files, desc="Processing MIDI"):
        for k in range(*transpose_range):
            try:
                bass, chords = parse_midi_file(path, transpose=k)
                if bass.size == 0:
                    continue
                Xn, Xc, Yn = slice_into_windows(bass, chords, T)
                if Xn is not None:
                    Xn_all.append(Xn)
                    Xc_all.append(Xc)
                    Yn_all.append(Yn)
            except Exception as e:
                print(f"{path} (+{k}): {e}")

    if not Xn_all:
        return None, None, None

    return (np.concatenate(Xn_all, axis=0),
            np.concatenate(Xc_all, axis=0),
            np.concatenate(Yn_all, axis=0))

# ───────────────────────── CLI TEST ───────────────────────── #

if __name__ == "__main__":
    Xn, Xc, Yn = build_training_dataset()
    if Xn is not None:
        print("Dataset shapes:")
        print("  X_notes :", Xn.shape)
        print("  X_chords:", Xc.shape)
        print("  y_notes :", Yn.shape)
    else:
        print("No data built.")

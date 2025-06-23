# ────────────────── GLOBAL SETTINGS & HYPERPARAMETERS ────────────────── #

# ─────────────——— DATA LOCATIONS ————───────── #
MIDI_FOLDER = "./midi_files/"   # folder with .mid files

# ─────────────——— TIME QUANTISATION ————───────── #
DEFAULT_BPM = 120
STEPS_PER_QUARTER = 4           # 16‑th‑note resolution
SEQUENCE_LENGTH   = 32          # slices of 32 time steps

# ─────────────——— BASS TOKEN RANGE ————───────── #
BASS_LOWEST_PITCH = 23          # MIDI note 23 (F1)
BASS_HIGHEST_PITCH = 62         # MIDI note 62 (D4)
BASS_REST_TOKEN = BASS_HIGHEST_PITCH - BASS_LOWEST_PITCH + 1   # = 40
BASS_NOTE_VOCAB_SIZE = BASS_REST_TOKEN + 1                     # = 41

# To keep downstream code unchanged ↓
REST_TOKEN      = BASS_REST_TOKEN
NOTE_VOCAB_SIZE = BASS_NOTE_VOCAB_SIZE

# ─────────────——— GUITAR CHORD VECTOR ————───────── #
GUITAR_LOWEST_PITCH = 40        # MIDI note 40 (E2)
GUITAR_HIGHEST_PITCH = 84       # MIDI note 84 (C6)
CHORD_VECTOR_SIZE = GUITAR_HIGHEST_PITCH - GUITAR_LOWEST_PITCH + 1   # = 45

# ─────────────——— MODEL ARCHITECTURE ————───────── #
NOTE_EMBED_DIM   = 16
CHORD_HIDDEN_DIM = 16
LSTM_UNITS       = 64

<<<<<<< HEAD
# ─────────────——— TRAINING ————───────── #
BATCH_SIZE   = 32
EPOCHS       = 15
=======
# Training
BATCH_SIZE = 32
EPOCHS = 15
>>>>>>> c85172d11d4521fb01369a673d092f62abc7cc18
LEARNING_RATE = 0.001

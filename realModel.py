import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

T = 32                   # sequence length (e.g., 2 measures of 16th notes in 4/4)
NOTE_VOCAB_SIZE = 46     # 32 pitch tokens + 1 'rest' or special token
CHORD_VECTOR_SIZE = 45   # 32-dim multi-hot chord vector
NOTE_EMBED_DIM = 16
CHORD_HIDDEN_DIM = 16
LSTM_UNITS = 64

# 3) BUILD THE REAL-TIME, SINGLE-STEP MODEL

# This model processes exactly 1 time step each call.
notes_in_rt = tf.keras.Input(
    batch_shape=(1, 1),  # batch=1, time=1
    dtype=tf.int32,
    name="notes_in_rt"
)
chords_in_rt = tf.keras.Input(
    batch_shape=(1, 1, CHORD_VECTOR_SIZE),  # batch=1, time=1, chord vec=32
    dtype=tf.float32,
    name="chords_in_rt"
)

# Reuse same layer names/dims so we can copy weights
note_embed_rt = layers.Embedding(
    input_dim=NOTE_VOCAB_SIZE,
    output_dim=NOTE_EMBED_DIM,
    name="note_embedding"
)(notes_in_rt)  # shape: (1, 1, NOTE_EMBED_DIM)

chord_dense_rt = layers.TimeDistributed(
    layers.Dense(CHORD_HIDDEN_DIM, activation='relu'),
    name="chord_dense"
)(chords_in_rt)  # shape: (1, 1, CHORD_HIDDEN_DIM)

combined_rt = layers.Concatenate(axis=-1)([note_embed_rt, chord_dense_rt])
# shape: (1, 1, NOTE_EMBED_DIM + CHORD_HIDDEN_DIM)

# Stateful LSTM: one step at a time
lstm_layer_rt = layers.LSTM(
    LSTM_UNITS,
    stateful=True,
    return_sequences=False,
    return_state=True,
    name="lstm"
)
lstm_out_rt, state_h_rt, state_c_rt = lstm_layer_rt(combined_rt)

note_logits_rt = layers.Dense(NOTE_VOCAB_SIZE, name="note_logits")(lstm_out_rt)
note_probs_rt = layers.Activation("softmax")(note_logits_rt)

rt_model = Model(
    inputs=[notes_in_rt, chords_in_rt],
    outputs=[note_probs_rt, state_h_rt, state_c_rt]
)

rt_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'  
    # In practice, we won't be training this single-step model
)

rt_model.summary()

#####################################
# 5) RUN REAL-TIME GENERATION (DEMO)
#####################################

# def sample_note(prob_dist, temperature=1.0):
#     """Sample an integer note token from a probability distribution with temperature."""
#     log_dist = np.log(prob_dist + 1e-9) / temperature
#     exp_dist = np.exp(log_dist)
#     softmax_dist = exp_dist / np.sum(exp_dist)
#     return np.random.choice(len(prob_dist), p=softmax_dist)

# # Reset the LSTM states at the start of generation
# rt_model.get_layer("lstm").reset_states()

# # We'll do a short generation of 16 steps
# generated_notes = []
# current_note = 0  # e.g., "rest" or "start" token

# # Make a dummy chord progression of length 16
# chord_prog = np.random.randint(0, 2, size=(16, CHORD_VECTOR_SIZE)).astype(np.float32)

# for t in range(16):
#     # Prepare model inputs: shape (1,1) for note, (1,1,32) for chord
#     note_input = np.array([[current_note]], dtype=np.int32)
#     chord_vec = chord_prog[t].reshape(1,1,CHORD_VECTOR_SIZE)  # shape (1,1,32)

#     # Predict
#     preds, h, c = rt_model.predict([note_input, chord_vec], verbose=0)
#     preds = preds[0]  # shape (NOTE_VOCAB_SIZE,)

#     # Sample next note
#     next_note = sample_note(preds, temperature=1.0)

#     generated_notes.append(next_note)
#     current_note = next_note

# print("Generated notes (tokens):", generated_notes)

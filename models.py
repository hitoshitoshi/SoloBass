# models.py

import tensorflow as tf
from tensorflow.keras import layers, Model

from config import (
    NOTE_VOCAB_SIZE,
    NOTE_EMBED_DIM,
    CHORD_HIDDEN_DIM,
    LSTM_UNITS,
    CHORD_VECTOR_SIZE,
    LEARNING_RATE
)

def build_unrolled_model():
    """
    Unrolled chord-conditional LSTM for offline training on sequences.
    Input: (batch, time), (batch, time, chord_vec)
    Output: (batch, time, note_vocab) with softmax over next note.
    """
    # Define inputs
    notes_in = tf.keras.Input(shape=(None,), name="notes_in")
    chords_in = tf.keras.Input(shape=(None, CHORD_VECTOR_SIZE), name="chords_in")
    
    # Note embedding layer
    note_embed = layers.Embedding(
        input_dim=NOTE_VOCAB_SIZE,
        output_dim=NOTE_EMBED_DIM,
        name="note_embedding"
    )(notes_in)  # (batch, time, NOTE_EMBED_DIM)
    
    # Chord projection via a dense layer applied at each time step
    chord_dense = layers.TimeDistributed(
        layers.Dense(CHORD_HIDDEN_DIM, activation='relu'),
        name="chord_dense"
    )(chords_in)  # (batch, time, CHORD_HIDDEN_DIM)
    
    # Concatenate the processed note and chord inputs
    combined = layers.Concatenate(axis=-1)([note_embed, chord_dense])
    # Now shape is: (batch, time, NOTE_EMBED_DIM + CHORD_HIDDEN_DIM)
    
    # Process with an LSTM layer
    lstm_out = layers.LSTM(LSTM_UNITS, return_sequences=True, name="lstm")(combined)
    # LSTM output shape: (batch, time, LSTM_UNITS)
    
    # Additional dense layers: stacking 5 TimeDistributed Dense layers
    x = lstm_out
    for i in range(1, 4):
        x = layers.TimeDistributed(
            layers.Dense(LSTM_UNITS, activation='relu'),
            name=f"dense_{i}"
        )(x)
    # x now passes through five non-linear transformations
    
    # Final projection to the note vocabulary and softmax activation
    note_logits = layers.TimeDistributed(
        layers.Dense(NOTE_VOCAB_SIZE),
        name="note_logits"
    )(x)
    note_probs = layers.Activation("softmax")(note_logits)
    
    # Build and compile the model
    model = Model([notes_in, chords_in], note_probs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_single_step_model():
    """
    Real-time model: processes exactly 1 time step per call using a stateful LSTM.
    Adds five Dense layers after the LSTM to deepen the transformation before final output.
    """
    # Define inputs with fixed batch=1 and time=1 dimensions
    notes_in_rt = tf.keras.Input(
        batch_shape=(1, 1),
        dtype=tf.int32,
        name="notes_in_rt"
    )
    chords_in_rt = tf.keras.Input(
        batch_shape=(1, 1, CHORD_VECTOR_SIZE),
        dtype=tf.float32,
        name="chords_in_rt"
    )
    
    # Embedding for notes
    note_embed_rt = layers.Embedding(
        input_dim=NOTE_VOCAB_SIZE,
        output_dim=NOTE_EMBED_DIM,
        name="note_embedding"
    )(notes_in_rt)  # (1, 1, NOTE_EMBED_DIM)
    
    # Chord projection with a dense layer applied using TimeDistributed
    chord_dense_rt = layers.TimeDistributed(
        layers.Dense(CHORD_HIDDEN_DIM, activation='relu'),
        name="chord_dense"
    )(chords_in_rt)  # (1, 1, CHORD_HIDDEN_DIM)
    
    # Concatenate the note and chord processing streams
    combined_rt = layers.Concatenate(axis=-1)([note_embed_rt, chord_dense_rt])
    
    # Stateful LSTM processes one timestep
    lstm_layer_rt = layers.LSTM(
        LSTM_UNITS,
        stateful=True,
        return_sequences=False,
        return_state=True,
        name="lstm"
    )
    lstm_out_rt, state_h_rt, state_c_rt = lstm_layer_rt(combined_rt)
    # lstm_out_rt shape: (1, LSTM_UNITS)
    
    # Additional dense layers (stacking 5 Dense layers with ReLU activations)
    x = lstm_out_rt
    for i in range(1, 4):
        x = layers.Dense(LSTM_UNITS, activation='relu', name=f"dense_{i}")(x)
    
    # Final projection for note prediction with softmax activation
    note_logits_rt = layers.Dense(NOTE_VOCAB_SIZE, name="note_logits")(x)
    note_probs_rt = layers.Activation("softmax")(note_logits_rt)
    
    # Build and compile the real-time model
    rt_model = Model(
        inputs=[notes_in_rt, chords_in_rt],
        outputs=[note_probs_rt, state_h_rt, state_c_rt]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    rt_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy'
    )
    return rt_model

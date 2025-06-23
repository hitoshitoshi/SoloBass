import time
import numpy as np
import mido
import tensorflow as tf
import fluidsynth

from config import LOWEST_PITCH, CHORD_VECTOR_SIZE, NOTE_VOCAB_SIZE, REST_TOKEN
from models import build_unrolled_model, build_single_step_model

def sample_note(prob_dist, temperature=1.0):
    """Randomly sample a note token from a probability distribution."""
    log_dist = np.log(prob_dist + 1e-9) / temperature
    exp_dist = np.exp(log_dist)
    softmax_dist = exp_dist / np.sum(exp_dist)
    return np.random.choice(range(len(prob_dist)), p=softmax_dist)

def main():
    WEIGHTS_PATH = "unrolled_lstm.weights.h5"
    
    # 1. Model Setup: Build unrolled model, load weights, then build single-step model and copy weights.
    unrolled_model = build_unrolled_model()
    unrolled_model.load_weights(WEIGHTS_PATH)
    
    rt_model = build_single_step_model()
    for rt_layer in rt_model.layers:
        try:
            source_layer = unrolled_model.get_layer(rt_layer.name)
            rt_layer.set_weights(source_layer.get_weights())
            print(f"Copied weights for layer '{rt_layer.name}'")
        except Exception as e:
            print(f"Skipping layer '{rt_layer.name}'")
    rt_model.get_layer("lstm").reset_states()

    # 2. Setup MIDI Input
    input_ports = mido.get_input_names()
    if input_ports:
        inport = mido.open_input(input_ports[1])
    else:
        print("No MIDI input ports available. Exiting.")
        return

    # 3. Setup FluidSynth for output.
    fs = fluidsynth.Synth()
    fs.start()
    sfid = fs.sfload("acoustic.sf2")
    fs.program_select(0, sfid, 0, 0)
    fs.setting('synth.gain', 1.5)

    # Initialize the chord vector (multi-hot), length = CHORD_VECTOR_SIZE.
    chord_vector = np.zeros((CHORD_VECTOR_SIZE,), dtype=np.float32)
    step_interval = 0.15  # Interval between note generation

    current_note = REST_TOKEN  # Start with REST
    last_played_note = None  # To track currently played note

    print("Starting real-time generation with MIDI input and FluidSynth output.")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            # 4. Update the chord vector from incoming MIDI messages.
            for msg in inport.iter_pending():
                if msg.type == 'note_on' and msg.velocity > 0:
                    idx = msg.note - LOWEST_PITCH
                    if 0 <= idx < CHORD_VECTOR_SIZE:
                        chord_vector[idx] = 1.0
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    idx = msg.note - LOWEST_PITCH
                    if 0 <= idx < CHORD_VECTOR_SIZE:
                        chord_vector[idx] = 0.0

            # 5. Generate next note using the current chord vector.
            chord_input = chord_vector.reshape((1, 1, CHORD_VECTOR_SIZE))
            note_input = np.array([[current_note]], dtype=np.int32)
            preds, _, _ = rt_model.predict([note_input, chord_input], verbose=0)
            preds = preds[0]  # shape: (NOTE_VOCAB_SIZE,)
            next_note = sample_note(preds, temperature=1.0)

            # If next_note is the same as last_played_note, sustain it
            if next_note != last_played_note:
                # Turn off the last played note if it's different
                if last_played_note is not None and last_played_note != REST_TOKEN:
                    fs.noteoff(0, last_played_note + LOWEST_PITCH)
                
                # Play new note if it's not a rest
                if next_note != REST_TOKEN:
                    midi_pitch = next_note + LOWEST_PITCH
                    fs.noteon(0, midi_pitch, 64)

                # Update last played note
                last_played_note = next_note

            # Update the current note for the next step
            current_note = next_note

            # Wait until next step.
            time.sleep(step_interval)
    except KeyboardInterrupt:
        print("Real-time generation stopped by user.")
    finally:
        inport.close()
        fs.delete()

if __name__ == "__main__":
    main()

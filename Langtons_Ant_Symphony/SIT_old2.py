import argparse
import numpy as np
import wave
import os

def generate_sine_wave(freq, duration, amplitude, sample_rate):
    """Generate a sine wave for a given frequency."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return amplitude * np.sin(2 * np.pi * freq * t)

def create_transition(freq_start, freq_end, duration, amplitude, sample_rate, num_samples):
    """Create a smooth transition between two frequencies using linear frequency interpolation."""
    t = np.linspace(0, duration, num_samples, False)
    frequencies = np.linspace(freq_start, freq_end, num_samples)
    return amplitude * np.sin(2 * np.pi * frequencies * t)

def main():
    parser = argparse.ArgumentParser(description="Generate a series of sine wave tones with smooth transitions.")
    parser.add_argument('-d', '--duration', type=float, default=10.0, help="Total duration in seconds")
    parser.add_argument('-n', '--num_tones', type=int, default=5, help="Number of sub-tones to generate")
    parser.add_argument('-s', '--sample_rate', type=int, default=44100, help="Sample rate in Hz")
    args = parser.parse_args()

    # Parameters
    total_duration = args.duration
    num_tones = args.num_tones
    sample_rate = args.sample_rate
    amplitude = 0.5  # Amplitude of the wave

    # Calculate samples
    total_samples = int(total_duration * sample_rate)
    sub_duration = total_duration / num_tones
    sub_samples = int(sub_duration * sample_rate)
    transition_duration = 0.1  # Transition duration in seconds
    transition_samples = int(transition_duration * sample_rate)

    # Generate list of frequencies
    frequencies = [np.random.uniform(200, 2000) for _ in range(num_tones)]
    print(f"Generated frequencies: {frequencies}")

    # Create an array for the whole audio
    audio = np.zeros(total_samples)

    # Initial tone
    start = 0
    audio[start:start + sub_samples] = generate_sine_wave(frequencies[0], sub_duration, amplitude, sample_rate)
    start += sub_samples - transition_samples  # Adjust start for next transition

    # Create transitions between tones
    for i in range(1, num_tones):
        # Transition part
        transition = create_transition(
            frequencies[i-1], 
            frequencies[i], 
            transition_duration, 
            amplitude, 
            sample_rate, 
            transition_samples
        )
        # Ensure the transition fits into the audio array
        audio[start:start + transition_samples] = transition
        start += transition_samples
        
        # Add remaining part of the tone before the next transition
        if i < num_tones - 1:
            tone_samples = sub_samples - transition_samples
            audio[start:start + tone_samples] = generate_sine_wave(frequencies[i], tone_samples / sample_rate, amplitude, sample_rate)
            start += tone_samples

    # Normalize audio
    max_abs_value = np.max(np.abs(audio))
    if max_abs_value > 1.0:
        audio = audio / max_abs_value

    # Save the audio to WAV file
    output_filename = "sine_tones.wav"
    with wave.open(output_filename, 'w') as wav_file:
        wav_file.setparams((1, 2, sample_rate, total_samples, "NONE", "not compressed"))
        audio_16bit = np.int16(audio * 32767)
        wav_file.writeframes(audio_16bit.tobytes())

    print(f"Generated audio file: {output_filename}")

if __name__ == "__main__":
    main()
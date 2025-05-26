import argparse
import numpy as np
import wave
import os

def generate_sine_wave(freq, duration, amplitude, sample_rate):
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return amplitude * np.sin(2 * np.pi * freq * t)

def interpolate_sine_waves(wave1, wave2, transition_length, total_length):
    """Interpolate smoothly between two sine waves with length adjustments."""
    if len(wave1) != len(wave2):
        min_length = min(len(wave1), len(wave2))
        wave1, wave2 = wave1[:min_length], wave2[:min_length]

    transition = np.linspace(0, 1, transition_length)
    interpolated = wave1[-transition_length:] * (1 - transition) + wave2[:transition_length] * transition
    result = np.concatenate([wave1[:-transition_length], interpolated, wave2[transition_length:]])
    return result[:total_length]  # Adjust to match the desired length

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
    amplitude = 0.5

    total_samples = int(total_duration * sample_rate)
    sub_duration = total_duration / num_tones
    transition_duration = 0.2  # Duration for each transition in seconds
    transition_samples = int(transition_duration * sample_rate)
    sub_samples = int(sub_duration * sample_rate)

    def random_freq():
        return np.random.uniform(200, 2000)  # Random frequency between 200 Hz and 2000 Hz

    # Generate list of frequencies
    frequencies = [random_freq() for _ in range(num_tones)]
    print(f"Generated frequencies: {frequencies}")

    # Create an array for the whole audio
    audio = np.zeros(total_samples)

    # Generate waves
    waves = [generate_sine_wave(freq, sub_duration, amplitude, sample_rate) for freq in frequencies]

    # Initial tone
    audio[:sub_samples] = waves[0]

    # Interpolate between tones
    for i in range(1, num_tones):
        start_idx = i * sub_samples
        end_idx = (i + 1) * sub_samples
        audio[start_idx:end_idx] = interpolate_sine_waves(waves[i-1], waves[i], transition_samples, sub_samples)

    # Normalize audio
    audio = audio / np.max(np.abs(audio))

    # Save the audio to WAV file
    output_filename = "sine_tones.wav"
    with wave.open(output_filename, 'w') as wav_file:
        wav_file.setparams((1, 2, sample_rate, total_samples, "NONE", "not compressed"))
        audio_16bit = np.int16(audio * 32767)
        wav_file.writeframes(audio_16bit.tobytes())

    print(f"Generated audio file: {output_filename}")

if __name__ == "__main__":
    main()
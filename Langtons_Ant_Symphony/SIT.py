import argparse
import numpy as np
import wave
import os

def generate_continuous_tone_sequence(frequencies, durations, amplitude, sample_rate):
    """Generate a continuous sequence of tones with smooth transitions throughout."""
    total_samples = int(sum(durations) * sample_rate)
    audio = np.zeros(total_samples)
    
    current_sample = 0
    phase = 0  # Keep track of phase to prevent discontinuities
    
    for i in range(len(frequencies) - 1):
        # Calculate number of samples for this segment
        segment_samples = int(durations[i] * sample_rate)
        
        # Create time array for this segment
        t = np.arange(segment_samples) / sample_rate
        
        # Calculate frequency for each sample (linear interpolation)
        freq_t = np.linspace(frequencies[i], frequencies[i + 1], segment_samples)
        
        # Calculate phase increment for each sample
        phase_increment = 2 * np.pi * freq_t / sample_rate
        
        # Accumulate phase
        phase_array = phase + np.cumsum(phase_increment)
        phase = phase_array[-1]  # Save final phase for continuity
        
        # Generate waveform
        segment = amplitude * np.sin(phase_array)
        
        # Apply subtle envelope to prevent clicks (2ms fade)
        fade_samples = int(0.002 * sample_rate)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            segment[:fade_samples] *= fade_in
            segment[-fade_samples:] *= fade_out
        
        # Add to main audio array
        audio[current_sample:current_sample + segment_samples] = segment
        current_sample += segment_samples
    
    return audio

def main():
    parser = argparse.ArgumentParser(description="Generate a series of sine wave tones with smooth transitions.")
    parser.add_argument('-d', '--duration', type=float, default=10.0, help="Total duration in seconds")
    parser.add_argument('-n', '--num_tones', type=int, default=5, help="Number of tones to generate")
    parser.add_argument('-s', '--sample_rate', type=int, default=44100, help="Sample rate in Hz")
    args = parser.parse_args()

    # Parameters
    total_duration = args.duration
    num_tones = args.num_tones
    sample_rate = args.sample_rate
    amplitude = 0.5

    # Generate random frequencies
    frequencies = [np.random.uniform(200, 2000) for _ in range(num_tones)]
    
    # Generate random durations that sum to total_duration
    raw_durations = np.random.uniform(0.5, 2.0, num_tones - 1)  # -1 because we need one less duration than frequencies
    durations = (raw_durations / np.sum(raw_durations)) * total_duration
    
    print(f"Generated frequencies: {frequencies}")
    print(f"Generated durations: {durations}")

    # Generate audio
    audio = generate_continuous_tone_sequence(frequencies, durations, amplitude, sample_rate)

    # Normalize audio
    max_abs_value = np.max(np.abs(audio))
    if max_abs_value > 1.0:
        audio = audio / max_abs_value

    # Save the audio to WAV file
    output_filename = "sine_tones.wav"
    with wave.open(output_filename, 'w') as wav_file:
        wav_file.setparams((1, 2, sample_rate, len(audio), "NONE", "not compressed"))
        audio_16bit = np.int16(audio * 32767)
        wav_file.writeframes(audio_16bit.tobytes())

    print(f"Generated audio file: {output_filename}")

if __name__ == "__main__":
    main()
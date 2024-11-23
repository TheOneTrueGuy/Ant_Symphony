import argparse
import numpy as np
import wave
from collections import defaultdict

class SoundAnt:
    def __init__(self, grid_size, rule):
        self.pos = (grid_size[0] // 2, grid_size[1] // 2)  # Start at center
        self.dir = 0  # 0: north, 1: east, 2: south, 3: west
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        self.rule = rule
        self.grid_size = grid_size

    def move(self, grid):
        current_color = grid[self.pos]
        
        # Turn based on rule
        if self.rule[current_color] == 'R':
            self.dir = (self.dir + 1) % 4
        else:
            self.dir = (self.dir - 1) % 4

        # Change color of current position
        grid[self.pos] = (current_color + 1) % len(self.rule)

        # Move forward
        dx, dy = self.directions[self.dir]
        new_x = (self.pos[0] + dx) % self.grid_size[0]  # Wrap around grid
        new_y = (self.pos[1] + dy) % self.grid_size[1]
        self.pos = (new_x, new_y)

        return self.pos

def generate_ant_symphony(args):
    # Create grid and initialize ants
    grid = defaultdict(int)
    grid_size = (100, 100)  # Grid size for frequency and amplitude resolution
    ants = [SoundAnt(grid_size, args.rule) for _ in range(args.num_ants)]
    
    # Calculate frequency and amplitude scales
    freq_scale = np.exp(np.linspace(np.log(args.min_freq), np.log(args.max_freq), grid_size[0]))
    amp_scale = np.linspace(args.min_amp, args.max_amp, grid_size[1])
    
    # Initialize audio array
    sample_rate = 44100
    total_samples = int(args.duration * sample_rate)
    audio = np.zeros(total_samples)
    
    # Number of samples per ant move
    samples_per_move = int(sample_rate / args.moves_per_second)
    
    # Generate audio
    phase = 0
    current_sample = 0
    
    while current_sample < total_samples:
        # Get current frequency and amplitude for each ant
        segment = np.zeros(min(samples_per_move, total_samples - current_sample))
        
        for ant in ants:
            # Move ant and get new position
            pos = ant.move(grid)
            
            # Map position to frequency and amplitude
            freq = freq_scale[pos[0]]
            amp = amp_scale[pos[1]]
            
            # Generate samples for this segment
            t = np.arange(len(segment)) / sample_rate
            phase_increment = 2 * np.pi * freq / sample_rate
            phase_array = phase + np.cumsum(np.full(len(segment), phase_increment))
            segment += amp * np.sin(phase_array)
        
        # Update phase for continuity
        phase = phase_array[-1] % (2 * np.pi)
        
        # Normalize segment to prevent clipping
        if len(ants) > 1:
            segment /= len(ants)
        
        # Add segment to audio
        audio[current_sample:current_sample + len(segment)] = segment
        current_sample += len(segment)
    
    return audio, sample_rate

def main():
    parser = argparse.ArgumentParser(description="Generate music using Langton's Ants on a frequency-amplitude grid")
    parser.add_argument('--num_ants', type=int, default=3, help='Number of ants')
    parser.add_argument('--rule', type=str, default='RL', help='Rule for ant movement (e.g., RL, RLR, LLRR)')
    parser.add_argument('--min_freq', type=float, default=200, help='Minimum frequency in Hz')
    parser.add_argument('--max_freq', type=float, default=2000, help='Maximum frequency in Hz')
    parser.add_argument('--min_amp', type=float, default=0.1, help='Minimum amplitude (0-1)')
    parser.add_argument('--max_amp', type=float, default=0.5, help='Maximum amplitude (0-1)')
    parser.add_argument('--duration', type=float, default=30, help='Duration in seconds')
    parser.add_argument('--moves_per_second', type=float, default=10, help='Ant moves per second')
    args = parser.parse_args()

    # Generate audio
    print(f"Generating {args.duration} seconds of ant symphony...")
    print(f"Using {args.num_ants} ants with rule: {args.rule}")
    audio, sample_rate = generate_ant_symphony(args)

    # Save to WAV file
    output_filename = "ant_symphony.wav"
    with wave.open(output_filename, 'w') as wav_file:
        wav_file.setparams((1, 2, sample_rate, len(audio), "NONE", "not compressed"))
        audio_16bit = np.int16(audio * 32767)
        wav_file.writeframes(audio_16bit.tobytes())

    print(f"Generated audio file: {output_filename}")

if __name__ == "__main__":
    main()

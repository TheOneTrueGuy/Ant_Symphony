import pygame
import sys
import argparse
import cv2
import numpy as np
from datetime import datetime
import wave
from collections import defaultdict

# Initialize Pygame
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1)

# Set up some constants
WIDTH, HEIGHT = 800, 800
CELL_SIZE = 5
SPEED = 300  # Lower numbers mean faster
SAMPLE_RATE = 44100

# Colors for the ants
ANT_COLORS = {
    0: (255, 0, 0),   # Red
    1: (0, 255, 0),   # Green
    2: (0, 0, 255),   # Blue
    3: (255, 255, 0), # Yellow
    4: (255, 0, 255), # Magenta
    5: (0, 255, 255), # Cyan
}

class Ant:
    def __init__(self, x, y, color_index, instructions):
        self.x = x
        self.y = y
        self.direction = 0  # 0: right, 1: up, 2: left, 3: down
        self.color = ANT_COLORS[color_index % len(ANT_COLORS)]
        self.instructions = instructions
        self.num_states = len(instructions)
        self.directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # right, up, left, down

    def move(self, grid):
        # Get current cell state
        cell_state = grid[self.x][self.y]
        
        # Use modulo to wrap state index to valid range for this ant's rules
        state_index = cell_state % self.num_states
        
        # Get instruction for current state
        instruction = self.instructions[state_index]

        # Turn left or right based on instruction
        if instruction == 'L':
            self.direction = (self.direction - 1) % 4
        else:  # instruction == 'R'
            self.direction = (self.direction + 1) % 4

        # Update cell state
        next_state = (cell_state + 1) % self.num_states
        grid[self.x][self.y] = next_state

        # Move forward
        dx, dy = self.directions[self.direction]
        self.x = (self.x + dx) % (WIDTH // CELL_SIZE)
        self.y = (self.y + dy) % (HEIGHT // CELL_SIZE)

        return self.x, self.y

def generate_audio_segment(ants, freq_scale, amp_scale, samples_per_frame):
    segment = np.zeros(samples_per_frame)
    
    # Generate audio based on ant positions only
    for ant in ants:
        # Map ant's x position to frequency
        freq = freq_scale[ant.x]
        # Map ant's y position to amplitude
        amp = amp_scale[ant.y]
        
        t = np.arange(samples_per_frame) / SAMPLE_RATE
        segment += amp * np.sin(2 * np.pi * freq * t)
    
    # Normalize to prevent clipping
    if np.max(np.abs(segment)) > 0:
        segment /= np.max(np.abs(segment))
    
    # Convert to 16-bit integer
    audio_16bit = np.int16(segment * 32767)
    return audio_16bit

def main():
    parser = argparse.ArgumentParser(description='Langton\'s Ants A/V Generator')
    parser.add_argument('--ants', type=str, default='RL',
                      help='Comma-separated rule strings for each ant (default: "RL")')
    parser.add_argument('--num_ants', type=int, default=2,
                      help='Number of ants to simulate (default: 2)')
    parser.add_argument('--min_freq', type=float, default=200,
                      help='Minimum frequency in Hz (default: 200)')
    parser.add_argument('--max_freq', type=float, default=2000,
                      help='Maximum frequency in Hz (default: 2000)')
    parser.add_argument('--min_amp', type=float, default=0.1,
                      help='Minimum amplitude 0-1 (default: 0.1)')
    parser.add_argument('--max_amp', type=float, default=0.5,
                      help='Maximum amplitude 0-1 (default: 0.5)')
    args = parser.parse_args()

    # Process the --ants argument into a list of rule strings
    rule_strings = [s.strip() for s in args.ants.split(',')]
    if not rule_strings:
        rule_strings = ['RL']
    
    # Validate rule strings
    valid_rules = []
    for rule in rule_strings:
        if not all(c in ('L', 'R') for c in rule):
            print(f"Warning: Invalid rule string '{rule}'. Using default 'RL'.")
            valid_rules.append('RL')
        else:
            valid_rules.append(rule)
    
    # Take the first 'num_ants' rules, filling with 'RL' if needed
    ant_rules = valid_rules[:args.num_ants]
    if len(ant_rules) < args.num_ants:
        ant_rules += ['RL'] * (args.num_ants - len(ant_rules))

    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Langton's Ants A/V")
    clock = pygame.time.Clock()

    # Initialize video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f'langtons_ants_av_{timestamp}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, 60.0, (WIDTH, HEIGHT))
    print(f"Recording to: {video_filename}")

    # Initialize audio file
    audio_filename = f'langtons_ants_av_{timestamp}.wav'
    audio_frames = []

    # Initialize the grid
    grid = [[0 for _ in range(HEIGHT // CELL_SIZE)] for _ in range(WIDTH // CELL_SIZE)]

    # Calculate frequency and amplitude scales
    freq_scale = np.exp(np.linspace(np.log(args.min_freq), np.log(args.max_freq), WIDTH // CELL_SIZE))
    amp_scale = np.linspace(args.min_amp, args.max_amp, HEIGHT // CELL_SIZE)

    # Create ants
    ants = []
    grid_width = WIDTH // CELL_SIZE
    grid_height = HEIGHT // CELL_SIZE
    center_x = grid_width // 2
    center_y = grid_height // 2
    spacing = 5
    
    for i in range(args.num_ants):
        offset = i * spacing
        start_x = (center_x + offset) % grid_width
        start_y = (center_y + offset) % grid_height
        ants.append(Ant(start_x, start_y, i, ant_rules[i]))
        print(f"Ant {i} starting at ({start_x}, {start_y}) with rules {ant_rules[i]}")

    # Calculate samples per frame
    samples_per_frame = int(SAMPLE_RATE / 60)  # 60 fps video

    # Main loop
    running = True
    frame_count = 0
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

            # Fill the screen with white
            screen.fill((255, 255, 255))

            # Draw the grid
            for x in range(WIDTH // CELL_SIZE):
                for y in range(HEIGHT // CELL_SIZE):
                    cell_state = grid[x][y]
                    if cell_state != 0:
                        color = ANT_COLORS[cell_state % len(ANT_COLORS)]
                        pygame.draw.rect(screen, color,
                                       (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

            # Draw and update all ants
            for ant in ants:
                pygame.draw.rect(screen, ant.color,
                               (ant.x * CELL_SIZE, ant.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                ant.move(grid)

            # Generate and play audio for this frame
            audio_segment = generate_audio_segment(ants, freq_scale, amp_scale, samples_per_frame)
            audio_frames.append(audio_segment)
            
            # Create a sound object and play it
            sound = pygame.sndarray.make_sound(audio_segment)
            sound.play()

            # Update the display
            pygame.display.flip()
            
            # Capture frame for video
            frame_count += 1
            if frame_count % 2 == 0:  # Capture every other frame
                frame_string = pygame.image.tostring(screen, 'RGB')
                frame = np.frombuffer(frame_string, dtype=np.uint8)
                frame = frame.reshape((HEIGHT, WIDTH, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            
            if frame_count % 100 == 0:
                print(f"Frames captured: {frame_count//2}")
                
            clock.tick(60)

    finally:
        print(f"Total frames captured: {frame_count//2}")
        
        # Save audio file
        audio_data = np.concatenate(audio_frames)
        with wave.open(audio_filename, 'w') as wav_file:
            wav_file.setparams((1, 2, SAMPLE_RATE, len(audio_data), "NONE", "not compressed"))
            audio_16bit = np.int16(audio_data * 32767)
            wav_file.writeframes(audio_16bit.tobytes())
        
        # Cleanup
        out.release()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main() 
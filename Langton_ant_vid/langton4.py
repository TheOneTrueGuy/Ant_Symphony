import pygame
import sys
import argparse
import cv2
import numpy as np
from datetime import datetime

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 800, 800
CELL_SIZE = 5
SPEED = 300  # Lower numbers mean faster

# Colors for the ants
ANT_COLORS = {
    0: (255, 0, 0),   # Red
    1: (0, 255, 0),   # Green
    2: (0, 0, 255),   # Blue
    3: (255, 255, 0), # Yellow
    4: (255, 0, 255), # Magenta
    5: (0, 255, 255), # Cyan
}

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Langton\'s Ants Simulation')
    parser.add_argument('--ants', type=str, default='RL',
                      help='Comma-separated rule strings for each ant (default: "RL"). ' +
                           'If there are fewer rules than ants, the remaining ants will use "RL". ' +
                           'If there are more rules than ants, the extra rules will be ignored.')
    parser.add_argument('--num_ants', type=int, default=2,
                      help='Number of ants to simulate (default: 2)')
    args = parser.parse_args()

    # Process the --ants argument into a list of rule strings
    rule_strings = [s.strip() for s in args.ants.split(',')]
    
    # Ensure there's at least one rule string
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
    pygame.display.set_caption("Langton's Ants")
    clock = pygame.time.Clock()

    # Initialize video writer with MP4 format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f'langtons_ants_{timestamp}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using MP4V codec which is more widely supported
    out = cv2.VideoWriter(video_filename, fourcc, 60.0, (WIDTH, HEIGHT))
    print(f"Recording to: {video_filename}")

    # Initialize the grid to store cell colors and states
    # Each cell will be a tuple of (state_index, color) where state_index indicates which rule to use
    grid = [[(0, None) for _ in range(HEIGHT // CELL_SIZE)] for _ in range(WIDTH // CELL_SIZE)]

    # Ant class
    class Ant:
        def __init__(self, x, y, color_index, instructions):
            self.x = x
            self.y = y
            self.direction = 0  # 0: right, 1: up, 2: left, 3: down
            self.color = ANT_COLORS[color_index % len(ANT_COLORS)]
            self.instructions = instructions
            self.num_states = len(instructions)  # Number of states this ant can handle

        def move(self):
            # Get current cell state
            cell_state, _ = grid[self.x][self.y]
            
            # Use modulo to wrap state index to valid range for this ant's rules
            state_index = cell_state % self.num_states
            
            # Get instruction for current state
            instruction = self.instructions[state_index]

            # Turn left or right based on instruction
            if instruction == 'L':
                self.direction = (self.direction - 1) % 4
            else:  # instruction == 'R'
                self.direction = (self.direction + 1) % 4

            # Update cell state (increment and wrap around to this ant's number of states)
            next_state = (cell_state + 1) % self.num_states
            grid[self.x][self.y] = (next_state, self.color)

            # Move forward
            if self.direction == 0: self.x = (self.x + 1) % (WIDTH // CELL_SIZE)
            elif self.direction == 1: self.y = (self.y - 1) % (HEIGHT // CELL_SIZE)
            elif self.direction == 2: self.x = (self.x - 1) % (WIDTH // CELL_SIZE)
            else: self.y = (self.y + 1) % (HEIGHT // CELL_SIZE)

    # Create ants
    ants = []
    grid_width = WIDTH // CELL_SIZE
    grid_height = HEIGHT // CELL_SIZE
    center_x = grid_width // 2
    center_y = grid_height // 2
    spacing = 5  # Space between ants
    
    for i in range(args.num_ants):
        # Calculate offset from center, placing ants in a diagonal pattern
        offset = i * spacing
        # Start position (diagonal pattern from center)
        start_x = (center_x + offset) % grid_width
        start_y = (center_y + offset) % grid_height
        # Create ant with validated position
        ants.append(Ant(start_x, start_y, i, ant_rules[i]))
        print(f"Ant {i} starting at ({start_x}, {start_y}) with rules {ant_rules[i]}")

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
                    cell_state, cell_color = grid[x][y]
                    if cell_state != 0:
                        color = cell_color if cell_color is not None else (0, 0, 0)
                        pygame.draw.rect(screen, color,
                                       (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

            # Draw and update all ants
            for ant in ants:
                # Draw the ant
                pygame.draw.rect(screen, ant.color,
                               (ant.x * CELL_SIZE, ant.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                ant.move()

            # Update the display
            pygame.display.flip()
            
            # Capture frame for video
            frame_count += 1
            if frame_count % 2 == 0:  # Capture every other frame to reduce load
                # Get the pygame surface as a string buffer and convert to numpy array
                frame_string = pygame.image.tostring(screen, 'RGB')
                frame = np.frombuffer(frame_string, dtype=np.uint8)
                frame = frame.reshape((HEIGHT, WIDTH, 3))
                # OpenCV uses BGR instead of RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            
            # Print progress
            if frame_count % 100 == 0:
                print(f"Frames captured: {frame_count//2}")
                
            clock.tick(SPEED)

    finally:
        print(f"Total frames captured: {frame_count//2}")
        # Cleanup
        out.release()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
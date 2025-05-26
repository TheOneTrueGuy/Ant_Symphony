import pygame
import argparse
import sys

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 800, 800
CELL_SIZE = 5
SPEED = 100  # Lower numbers mean faster

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
                      help='String of "L" and "R" instructions for each ant (default: "RL")')
    parser.add_argument('--num_ants', type=int, default=2,
                      help='Number of ants to simulate (default: 2)')
    args = parser.parse_args()

    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Langton's Ants")
    clock = pygame.time.Clock()

    # Initialize the grid to store cell colors (0 for unvisited, 1 for visited, -1 for special)
    # We'll use -1 to represent unvisited cells, and 0-5 to represent ant colors
    grid = [[-1 for _ in range(HEIGHT // CELL_SIZE)] for _ in range(WIDTH // CELL_SIZE)]

    # Ant class
    class Ant:
        def __init__(self, x, y, color_index, instructions):
            self.x = x
            self.y = y
            self.direction = 0  # 0: right, 1: up, 2: left, 3: down
            self.color = ANT_COLORS[color_index % len(ANT_COLORS)]
            self.instructions = instructions

        def move(self):
            # Get current cell color
            cell_color = grid[self.x][self.y]

            # If the cell is unvisited (-1), treat it as white (0)
            if cell_color == -1:
                current_color = 0
            else:
                current_color = 1  # Any color indicates a visited cell

            # Get instruction (L or R) based on cell color
            instruction = self.instructions[current_color]

            # Turn left or right
            if instruction == 'L':
                self.direction = (self.direction - 1) % 4
            else:
                self.direction = (self.direction + 1) % 4

            # Leave a trail of the ant's color
            grid[self.x][self.y] = self.color

            # Move forward
            if self.direction == 0: self.x = (self.x + 1) % (WIDTH // CELL_SIZE)
            elif self.direction == 1: self.y = (self.y - 1) % (HEIGHT // CELL_SIZE)
            elif self.direction == 2: self.x = (self.x - 1) % (WIDTH // CELL_SIZE)
            else: self.y = (self.y + 1) % (HEIGHT // CELL_SIZE)

    # Create ants
    ants = []
    for i in range(args.num_ants):
        # Start position (center of the grid)
        start_x = (WIDTH // CELL_SIZE) // 2 + i
        start_y = (HEIGHT // CELL_SIZE) // 2
        ants.append(Ant(start_x, start_y, i, args.ants))

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        # Fill the screen with white
        screen.fill((255, 255, 255))

        # Draw the grid with colored trails
        for x in range(WIDTH // CELL_SIZE):
            for y in range(HEIGHT // CELL_SIZE):
                cell_color = grid[x][y]
                if cell_color != -1:
                    pygame.draw.rect(screen, cell_color,
                                   (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Draw and update all ants
        for ant in ants:
            # Draw the ant
            pygame.draw.rect(screen, ant.color,
                           (ant.x * CELL_SIZE, ant.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            ant.move()

        # Update the display
        pygame.display.flip()
        clock.tick(SPEED)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
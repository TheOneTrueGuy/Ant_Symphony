from collections import defaultdict

def multi_color_langtons_ant(steps, rule, num_colors):
    # Example rule: "RLLR" for 4 colors, "RL" for 2 colors
    grid = defaultdict(int)  # 0 represents color[0], 1 represents color[1], etc.
    ant_pos = (0, 0)
    ant_dir = 0  # 0: north, 1: east, 2: south, 3: west
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W

    for _ in range(steps):
        current_color = grid[ant_pos]

        # Decide turn direction
        if rule[current_color] == 'R':
            ant_dir = (ant_dir + 1) % 4  # Turn right
        else:
            ant_dir = (ant_dir - 1) % 4  # Turn left

        # Change color of current position
        grid[ant_pos] = (current_color + 1) % len(rule)

        # Move ant forward
        dx, dy = directions[ant_dir]
        ant_pos = (ant_pos[0] + dx, ant_pos[1] + dy)

    return grid

# Example usage with 4 colors and rule "RLLR"
result = multi_color_langtons_ant(10000, "RLLR", 4)

# If you want to visualize or analyze, you would need to convert this grid to a visual representation or further process it.
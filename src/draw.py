import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def get_coordinates(num, n):
    """Helper function to convert cell number to grid coordinates."""
    row = (num - 1) // n
    col = (num - 1) % n
    if row % 2 == 0:
        x = col
    else:
        x = n - 1 - col
    y = row
    return x, y


def draw_snake(ax, start, end, n):
    """Draw a snake from start to end with a curved line."""
    x_start, y_start = get_coordinates(start, n)
    x_end, y_end = get_coordinates(end, n)
    y_end += 0.2
    y_start -= 0.2

    snake_body = patches.FancyArrowPatch(
        (x_start + 0.5, y_start + 0.5),
        (x_end + 0.5, y_end + 0.5),
        connectionstyle=f"arc3,rad={-0.2}",
        color="red",
        linewidth=4,
    )
    ax.add_patch(snake_body)

    # Optional: Add a "snake head" (circle at the snake's tail)
    ax.add_patch(patches.Circle((x_start + 0.5, y_start + 0.5), 0.15, color="red"))
    ax.add_patch(patches.Circle((x_start + 0.4, y_start + 0.55), 0.02, color="black"))
    ax.add_patch(patches.Circle((x_start + 0.6, y_start + 0.55), 0.02, color="black"))


def draw_ladder(ax, start, end, n):
    """Draw a ladder with two parallel lines and rungs."""
    x_start, y_start = get_coordinates(start, n)
    x_end, y_end = get_coordinates(end, n)
    y_end -= 0.2
    y_start += 0.2

    # Draw two vertical parallel lines for the ladder sides
    ax.plot(
        [x_start + 0.3, x_end + 0.3], [y_start + 0.5, y_end + 0.5], color="green", lw=3
    )
    ax.plot(
        [x_start + 0.7, x_end + 0.7], [y_start + 0.5, y_end + 0.5], color="green", lw=3
    )

    # Draw the ladder rungs (horizontal lines between the two sides)
    num_rungs = int(
        np.hypot(x_end - x_start, y_end - y_start) * 10
    )  # Adjust number of rungs
    for i in np.linspace(0, 1, num_rungs):
        rung_x_start = x_start + 0.3 + i * (x_end - x_start)
        rung_y_start = y_start + 0.5 + i * (y_end - y_start)
        ax.plot(
            [rung_x_start, rung_x_start + 0.4],
            [rung_y_start, rung_y_start],
            color="green",
            lw=2,
        )


def draw_board(ax, n, snakes, ladders):
    # Draw the grid
    for i in range(n + 1):
        ax.plot([0, n], [i, i], color="black")  # Horizontal lines
        ax.plot([i, i], [0, n], color="black")  # Vertical lines

    # Add numbers to the grid, starting from the bottom
    for row in range(n):
        for col in range(n):
            if row % 2 == 0:
                # Even rows: numbers increase left to right
                num = n * row + col + 1
            else:
                # Odd rows: numbers increase right to left
                num = n * (row + 1) - col

            # Calculate position of the text (upper-left corner of the cell)
            x_pos = col + 0.1  # Slightly offset to the right
            y_pos = row + 0.9  # Slightly offset downwards

            # Place the number in the upper-left corner of the cell
            ax.text(x_pos, y_pos, str(num), ha="left", va="top", fontsize=12)

    # Draw ladders
    for start, end in ladders:
        draw_ladder(ax, start, end, n)

    # Draw snakes
    for start, end in snakes:
        draw_snake(ax, start, end, n)


def animate_player_movement(n, snakes, ladders, path):
    """Animate the player moving along a path."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    draw_board(ax, n, snakes, ladders)

    # Draw the player as a blue circle that will move
    player_circle = patches.Circle((0.5, 0.5), 0.3, color="blue")
    ax.add_patch(player_circle)

    def update(frame):
        """Update the player's position on the board."""
        x, y = get_coordinates(path[frame], n)
        player_circle.center = (x + 0.5, y + 0.5)
        return (player_circle,)

    anim = FuncAnimation(
        fig, update, frames=len(path), interval=500, blit=True, repeat=False
    )

    return anim

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


COLORS = {
    "snake": "#EF476F",
    "snake_eyes": "#FFD166",
    "ladder": "#06D6A0",
    "grid": "#073B4C",
    "player": "#118AB2",
    "background": "#FAFEFF",
    "highlight": "#44C3EE",
    "goal": "#FFD166",
}


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
    ax.add_patch(
        patches.Circle((x_start + 0.5, y_start + 0.5), 0.15, color=COLORS["snake"])
    )
    ax.add_patch(
        patches.Circle(
            (x_start + 0.4, y_start + 0.55), 0.02, color=COLORS["snake_eyes"]
        )
    )
    ax.add_patch(
        patches.Circle(
            (x_start + 0.6, y_start + 0.55), 0.02, color=COLORS["snake_eyes"]
        )
    )


def draw_ladder(ax, start, end, n):
    """Draw a ladder with two parallel lines and rungs."""
    x_start, y_start = get_coordinates(start, n)
    x_end, y_end = get_coordinates(end, n)
    y_end -= 0.2
    y_start += 0.2

    # Draw two vertical parallel lines for the ladder sides
    ax.plot(
        [x_start + 0.3, x_end + 0.3],
        [y_start + 0.5, y_end + 0.5],
        color=COLORS["ladder"],
        lw=3,
    )
    ax.plot(
        [x_start + 0.7, x_end + 0.7],
        [y_start + 0.5, y_end + 0.5],
        color=COLORS["ladder"],
        lw=3,
    )

    # Draw the ladder rungs (horizontal lines between the two sides)
    num_rungs = int(abs(y_end - y_start) * 10)
    for i in np.linspace(0, 1, num_rungs):
        rung_x_start = x_start + 0.3 + i * (x_end - x_start)
        rung_y_start = y_start + 0.5 + i * (y_end - y_start)
        ax.plot(
            [rung_x_start, rung_x_start + 0.4],
            [rung_y_start, rung_y_start],
            color=COLORS["ladder"],
            lw=2,
        )


def draw_board(ax, n, snakes, ladders, good_squares):
    # for i in range(n + 1):
    #     ax.plot([0, n], [i, i], color=COLORS["grid"])
    #     ax.plot([i, i], [0, n], color=COLORS["grid"])

    for row in range(n):
        for col in range(n):
            if row % 2 == 0:
                num = n * row + col + 1
            else:
                num = n * (row + 1) - col

            x_pos = col + 0.1
            y_pos = row + 0.9

            ax.text(x_pos, y_pos, str(num), ha="left", va="top", fontsize=12)

    for good_square in good_squares:
        good = patches.Rectangle(
            get_coordinates(good_square, n),
            1,
            1,
            color=COLORS["goal"],
            linewidth=10,
            linestyle="-" if good_square == n**2 else (good_square, (1, 2)),
            fill=False,
            zorder=1,
            alpha=1,
            clip_on=True,
        )
        ax.add_patch(good)
        good.set_clip_path(good)

    for start, end in ladders.items():
        draw_ladder(ax, start, end, n)

    for start, end in snakes.items():
        draw_snake(ax, start, end, n)


def animate_player_movement(
    n: int,
    snakes: dict[int, int],
    ladders: dict[int, int],
    goal_squares: list[int],
    path: list[int],
):
    """Animate the player moving along a path."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_facecolor(COLORS["background"])
    ax.set_xticks(range(n + 1), ["" for _ in range(n + 1)])
    ax.xaxis.set_ticks_position("none")
    ax.set_yticks(range(n + 1), ["" for _ in range(n + 1)])
    ax.yaxis.set_ticks_position("none")
    ax.grid(True, color=COLORS["grid"], linestyle="-", linewidth=1)
    ax.set_aspect("equal")
    draw_board(ax, n, snakes, ladders, goal_squares)

    # Draw the player as a blue circle that will move
    PLAYER_WIDTH = 0.2
    player_circle = patches.Circle(
        (0.5, 0.5),
        PLAYER_WIDTH,
        color=COLORS["player"],
        animated=True,
        zorder=10,
    )
    ax.add_patch(player_circle)

    # Add highlight patches
    highlights = []
    for i in range(n):
        highlight = patches.Rectangle(
            (0, 0),
            1,
            1,
            color=COLORS["highlight"],
            zorder=2,
            visible=False,
            alpha=0.3,
            linewidth=2,
        )
        ax.add_patch(highlight)
        highlights.append(highlight)

    def update(frame):
        """Update the player's position on the board."""
        x, y = get_coordinates(path[frame], n)
        player_circle.center = (x + 0.5, y + 0.5)
        if frame < len(path) - 1:
            i = 0
            if path[frame + 1] in snakes:
                for s in snakes:
                    if path[frame] < s <= path[frame] + 4:
                        highlights[i].set_xy(get_coordinates(s, n))
                        highlights[i].set_color(COLORS["snake"])
                        highlights[i].set_visible(True)
                        i += 1
            elif path[frame + 1] in ladders:
                for s in ladders:
                    if path[frame] < s <= path[frame] + 4:
                        highlights[i].set_xy(get_coordinates(s, n))
                        highlights[i].set_color(COLORS["ladder"])
                        highlights[i].set_visible(True)
                        i += 1
            else:
                if path[frame] in ladders:
                    highlights[i].set_xy(get_coordinates(ladders[path[frame]], n))
                    highlights[i].set_color(COLORS["highlight"])
                    highlights[i].set_visible(True)
                    i += 1
                elif path[frame] in snakes:
                    highlights[i].set_xy(get_coordinates(snakes[path[frame]], n))
                    highlights[i].set_color(COLORS["highlight"])
                    highlights[i].set_visible(True)
                    i += 1
                else:
                    for s in range(path[frame] + 1, path[frame] + 5):
                        if s not in ladders and s not in snakes:
                            highlights[i].set_xy(get_coordinates(s, n))
                            highlights[i].set_color(COLORS["highlight"])
                            highlights[i].set_visible(True)
                            i += 1
            for j in range(i, 4):
                highlights[j].set_visible(False)
        else:
            for h in highlights:
                h.set_visible(False)

        return player_circle, *highlights

    anim = FuncAnimation(
        fig, update, frames=len(path), interval=500, blit=True, repeat=False
    )

    return anim

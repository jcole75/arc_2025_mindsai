"""
Terminal-based visualization for ARC tasks and grids using ANSI escape codes.
"""

# ANSI escape codes for background colors, plus a reset code.
# The mappings use xterm 256-color codes to approximate the official ARC palette.
ANSI_COLORS = {
    0: "\x1b[48;5;16m",  # Black (#000000)
    1: "\x1b[48;5;33m",  # Blue (#0074D9)
    2: "\x1b[48;5;196m",  # Red (#FF4136)
    3: "\x1b[48;5;46m",  # Green (#2ECC40)
    4: "\x1b[48;5;220m",  # Yellow (#FFDC00)
    5: "\x1b[48;5;248m",  # Grey (#AAAAAA)
    6: "\x1b[48;5;205m",  # Pink (#F012BE)
    7: "\x1b[48;5;208m",  # Orange (#FF851B)
    8: "\x1b[48;5;159m",  # Light Blue (#7FDBFF)
    9: "\x1b[48;5;88m",  # Maroon (#870C25)
}
RESET_CODE = "\x1b[0m"


def draw_grid_terminal(grid: list[list[int]], title: str = "") -> str:
    """
    Renders a single ARC grid as a string with ANSI colors for terminal display.

    Args:
        grid (list[list[int]]): The grid to visualize.
        title (str): An optional title to print above the grid.

    Returns:
        A formatted, multi-line string representing the colored grid.
    """
    if not grid or not isinstance(grid, list) or not isinstance(grid[0], list):
        return f"{title}\n[Invalid or Empty Grid]\n"

    header = f"{title}\n" if title else ""
    output_str = ""
    for row in grid:
        for cell in row:
            color = ANSI_COLORS.get(cell, ANSI_COLORS[5])  # Default to grey background
            # Use black text for light backgrounds for better readability
            text_color = "\x1b[30m" if cell in {4, 5, 6, 7, 8} else "\x1b[37m"
            output_str += f"{color}{text_color} {cell} {RESET_CODE}"
        output_str += "\n"
    return header + output_str


def visualize_task_terminal(task: dict, title: str = "ARC Task"):
    """
    Prints a full visualization of an ARC task (train and test pairs) to the terminal.

    Args:
        task (dict): The ARC task dictionary.
        title (str): An optional title for the entire task visualization.
    """
    print(f"\n--- {title} ---")

    # Visualize training examples
    for i, ex in enumerate(task.get("train", [])):
        print(f"\n--- Train Example {i + 1} ---")
        print("Input:")
        print(draw_grid_terminal(ex["input"]))
        print("Output:")
        print(draw_grid_terminal(ex["output"]))

    # Visualize test examples
    for i, ex in enumerate(task.get("test", [])):
        print(f"\n--- Test Example {i + 1} ---")
        print("Input:")
        print(draw_grid_terminal(ex["input"]))
        if "output" in ex:  # Handle cases where test output is known
            print("Output:")
            print(draw_grid_terminal(ex["output"]))

    print("-" * (len(title) + 8))

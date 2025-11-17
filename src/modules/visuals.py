import base64
from collections import Counter
import io
import json
import re

from matplotlib.colors import ListedColormap, Normalize
import matplotlib.pyplot as plt
import numpy as np


# --- Global Color Map for ARC Grids ---
# build a color map using ARC's standard colors
cmap_list = [
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
]
cmap_values = ListedColormap(cmap_list)
norm = Normalize(vmin=0, vmax=9)

# --- Utility Functions for Grid Manipulation ---


def list_to_grid(grid_list):
    """Converts a list of lists (representing a grid) back to a NumPy array."""
    return np.array(grid_list, dtype=int)


def remove_empty_arrays(array):
    """Removes empty strings or lists from an array/list."""
    return [x for x in array if x not in ["", " ", "  ", []]]


def remove_leading_and_trailing_spaces(array):
    """Removes leading and trailing spaces from all strings in a list."""
    return [x.strip() for x in array]


def string_to_int_array(string_array):
    """
    Converts a string of numbers (e.g., '22562') to a list of integers (e.g., [2, 2, 5, 6, 2]).
    Can handle a list of such strings.
    """
    if isinstance(string_array, list):
        ret_int_array = []
        for s_item in string_array:
            int_row = []
            for char in s_item:
                if char.isdigit():
                    int_row.append(int(char))
            if int_row:  # Only append if the row is not empty after cleaning
                ret_int_array.append(int_row)
        return ret_int_array
    else:  # Assume single string
        int_array = []
        for char in string_array:
            if char.isdigit():
                int_array.append(int(char))
        return int_array


def string_to_char_array(string_array):
    """
    Converts a string of numbers to a list of characters.
    Example: '22562' -> ['2', '2', '5', '6', '2']
    Can handle a list of such strings.
    """
    if isinstance(string_array, list):
        ret_char_array = []
        for s_item in string_array:
            char_row = [char for char in s_item]
            ret_char_array.append(char_row)
        return ret_char_array
    else:  # Assume single string
        return [char for char in string_array]


def convert_arc_board_string_to_array(board_string):
    """
    Converts a string representation of an ARC grid (e.g., "123 456 789") into
    a list of lists of integers. It attempts to handle inconsistent row lengths
    by adjusting rows to match the most common (modal) row length found.
    Assumes the input string contains only digit rows separated by spaces.
    """
    trimmed_string = str(board_string).strip()

    # Split the string by spaces, filter out empty parts
    parts = [part.strip() for part in trimmed_string.split(" ") if part.strip()]

    board_array = []
    for row_str in parts:
        # Clean each row by removing non-numeric characters and then convert to integers
        cleaned_row = "".join(ch for ch in row_str if ch.isdigit())
        if cleaned_row:
            try:
                num_row = list(map(int, cleaned_row))
                board_array.append(num_row)
            except ValueError:
                # Skip this potentially corrupted row
                continue

    if not board_array:
        # Return empty list if no valid rows are found
        return []

    # Calculate the modal length of the rows to handle potentially ragged arrays
    try:
        length_counts = Counter(len(row) for row in board_array)
        if not length_counts:  # Should not happen if board_array is not empty
            return []
        modal_length = length_counts.most_common(1)[0][0]
    except IndexError:
        return []
    except Exception:
        return []

    # Adjust each row to match the modal length
    adjusted_board_array = []
    for current_row in board_array:
        current_len = len(current_row)
        if current_len == modal_length:
            adjusted_board_array.append(current_row)
        elif current_len > modal_length:
            adjusted_board_array.append(current_row[:modal_length])  # Truncate if too long
        elif current_len < modal_length:
            if current_len > 0:
                # Pad with the last element if not empty
                adjusted_board_array.append(current_row + [current_row[-1]] * (modal_length - current_len))
            else:
                # If an empty row somehow gets through, pad with zeros
                adjusted_board_array.append([0] * modal_length)

    return adjusted_board_array


# --- Functions for Parsing ARC JSON and NLP Strings ---


def get_arc_arrays(json_data):
    """
    Extracts input and output grids (as lists of lists of integers) from an ARC task JSON.
    Can accept either a JSON string or a loaded dictionary.
    """
    json_obj = json.loads(json_data) if isinstance(json_data, str) else json_data

    train_inputs = []
    train_outputs = []
    test_inputs = []
    test_outputs = []

    for example in json_obj["train"]:
        train_inputs.append(example["input"])
        train_outputs.append(example["output"])

    for example in json_obj["test"]:
        test_inputs.append(example["input"])
        if "output" in example:  # Output might be missing in test cases
            test_outputs.append(example["output"])

    # Convert all elements to integers for consistency, handling potential floats
    try:
        train_inputs = [[[int(y) for y in x] for x in z] for z in train_inputs]
        train_outputs = [[[int(y) for y in x] for x in z] for z in train_outputs]
        test_inputs = [[[int(y) for y in x] for x in z] for z in test_inputs]
        test_outputs = [[[int(y) for y in x] for x in z] for z in test_outputs]
    except (TypeError, ValueError):
        # If conversion fails (e.g., non-numeric data), keep original type
        pass

    return train_inputs, train_outputs, test_inputs, test_outputs


def convert_nlp_arc_string_to_arc_arrays(nlp_string, grid_starting_row=4):
    """
    Converts a specific NLP string representation of an ARC task back into
    structured ARC arrays (list of lists of integers for inputs and outputs).
    This function expects a format like:
    "train input1 123 456 output1 789. test tinput1 112 233 toutput1 445 566."
    It parses out the labels and then converts the digit strings into grids.
    """
    if not isinstance(nlp_string, str):
        print(f"Error: Expected string for nlp_string, got {type(nlp_string)}. Cannot parse.")
        return [], [], [], []

    train_part = nlp_string.split("test")[0]
    test_part = nlp_string.split("test")[1] if "test" in nlp_string else ""

    train_pairs_raw = remove_empty_arrays(remove_leading_and_trailing_spaces(train_part.split(".")))
    test_pairs_raw = remove_empty_arrays(remove_leading_and_trailing_spaces(test_part.split(".")))

    train_input_strings = []
    train_output_strings = []
    for pair_str in train_pairs_raw:
        parts = pair_str.split("output")
        if len(parts) == 2:
            # Remove "inputX " prefix (e.g., "input1 ")
            train_input_strings.append(re.sub(r"^[a-z]+[0-9]+\s*", "", parts[0].strip()))
            # Remove "outputX " prefix (e.g., "output1 ")
            train_output_strings.append(re.sub(r"^[a-z]+[0-9]+\s*", "", parts[1].strip()))

    test_input_strings = []
    test_output_strings = []
    for pair_str in test_pairs_raw:
        parts = pair_str.split("toutput1")  # Assuming 'toutput1' is the delimiter for test outputs
        if len(parts) == 2:
            # Remove "tinputX " prefix (e.g., "tinput1 ")
            test_input_strings.append(re.sub(r"^[a-z]+[0-9]+\s*", "", parts[0].strip()))
            # Remove "toutputX " prefix (e.g., "toutput1 ")
            test_output_strings.append(re.sub(r"^[a-z]+[0-9]+\s*", "", parts[1].strip()))
        elif len(parts) == 1 and parts[0].strip():
            # Handle test input without an explicit output (e.g., for prediction tasks)
            test_input_strings.append(re.sub(r"^[a-z]+[0-9]+\s*", "", parts[0].strip()))
            test_output_strings.append("")  # Mark output as empty

    # Convert the processed strings into 2D integer arrays
    train_inputs = [convert_arc_board_string_to_array(s) for s in train_input_strings]
    train_outputs = [convert_arc_board_string_to_array(s) for s in train_output_strings]
    test_inputs = [convert_arc_board_string_to_array(s) for s in test_input_strings]
    test_outputs = [convert_arc_board_string_to_array(s) for s in test_output_strings]

    # Filter out any empty lists that might result from conversion errors
    train_inputs = [g for g in train_inputs if g]
    train_outputs = [g for g in train_outputs if g]
    test_inputs = [g for g in test_inputs if g]
    test_outputs = [g for g in test_outputs if g]

    return train_inputs, train_outputs, test_inputs, test_outputs


# --- Core Plotting Functions ---


def draw_grid(ax, grid, line_width):
    """Draws grid lines on a matplotlib Axes object."""
    grid = np.array(grid)  # Ensure it's a NumPy array

    # Handle empty or malformed grids gracefully
    if grid.shape[0] == 0 or grid.shape[1] == 0:
        return

    # Calculate positions for grid lines
    yy, xx = np.arange(grid.shape[0]) + 0.5, np.arange(grid.shape[1]) + 0.5

    # Draw horizontal and vertical lines
    ax.hlines(y=yy, xmin=-0.5, xmax=grid.shape[1] - 0.5, color="gray", linewidth=line_width)
    ax.vlines(x=xx, ymin=-0.5, ymax=grid.shape[0] - 0.5, color="gray", linewidth=line_width)


def plot_grid(ax, grid, title="", gridlines=True):
    """
    Plots a single ARC grid on a given matplotlib Axes object.
    Includes title and optional grid lines.
    """
    # Ensure grid is a NumPy array for consistent plotting
    grid = np.array(grid, dtype=int)

    ax.imshow(grid, cmap=cmap_values, norm=norm, origin="upper")
    ax.set_title(title)
    ax.set_xticks([])  # Hide x-axis ticks
    ax.set_yticks([])  # Hide y-axis ticks
    if gridlines:
        draw_grid(ax, grid, 0.5)
    ax.set_aspect("equal", adjustable="box")  # Ensure cells are square


def plot_riddle_two_rows(
    train_input_grids,
    train_output_grids,
    test_input_grids,
    test_output_grids,
    gridlines=False,
    return_base64=False,
    max_train_pairs=None,
):
    """
    Plots ARC grids for train and test examples in a two-row layout.
    Input grids are in the first row, corresponding output grids in the second row.

    Args:
        train_input_grids (list[list[list[int]]]): List of 2D input grids for training.
        train_output_grids (list[list[list[int]]]): List of 2D output grids for training.
        test_input_grids (list[list[list[int]]]): List of 2D input grids for testing.
        test_output_grids (list[list[list[int]]]): List of 2D output grids for testing.
        gridlines (bool): If True, draws grid lines on the plots.
        return_base64 (bool): If True, returns a base64 encoded PNG image string instead of displaying.
        max_train_pairs (int): Maximum number of training pairs to display (None = show all).

    Returns:
        matplotlib.figure.Figure or str: The matplotlib Figure object if `return_base64` is False,
                                        or a base64 encoded image string if `return_base64` is True.
                                        Returns None if no grids are provided.
    """
    # Defensive programming: ensure all inputs are lists of lists of lists for consistent processing
    train_input_grids = [np.array(g, dtype=int) for g in train_input_grids if np.array(g).size > 0]
    train_output_grids = [np.array(g, dtype=int) for g in train_output_grids if np.array(g).size > 0]
    test_input_grids = [np.array(g, dtype=int) for g in test_input_grids if np.array(g).size > 0]
    test_output_grids = [np.array(g, dtype=int) for g in test_output_grids if np.array(g).size > 0]

    # Create placeholders for missing test outputs if test inputs exist
    if test_input_grids and not test_output_grids:
        # Fill with dummy grids of the same shape as inputs, using background color 0
        test_output_grids = [np.full_like(inp, 0) for inp in test_input_grids]
    elif not test_input_grids and test_output_grids:
        # If outputs exist without inputs (unlikely for ARC), create dummy inputs
        test_input_grids = [np.full_like(outp, 0) for outp in test_output_grids]

    n_train_pairs = len(train_input_grids)
    n_test_pairs = len(test_input_grids)

    # Only apply limit if specified
    if max_train_pairs is not None and n_train_pairs > max_train_pairs:
        print(f"Note: Limiting display to first {max_train_pairs} of {n_train_pairs} training pairs.")
        train_input_grids = train_input_grids[:max_train_pairs]
        train_output_grids = train_output_grids[:max_train_pairs]
        n_train_pairs = max_train_pairs

    total_cols = n_train_pairs + n_test_pairs

    if total_cols == 0:
        print("No grids to plot.")
        return None

    # Dynamic figure sizing - no maximum width limit
    fig_width = total_cols * 3.5  # Allow any width needed
    fig, axs = plt.subplots(2, total_cols, figsize=(fig_width, 7))

    # Handle cases where axs might not be a 2D array (e.g., only one column)
    if total_cols == 1:
        axs = np.array([axs]).reshape(2, 1)  # Reshape to 2 rows, 1 column for consistent indexing

    plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.05, right=0.95, top=0.9, bottom=0.1)

    # Plotting training data
    for r in range(n_train_pairs):
        plot_grid(axs[0, r], train_input_grids[r], title=f"Train Input {r + 1}", gridlines=gridlines)
        plot_grid(axs[1, r], train_output_grids[r], title=f"Train Output {r + 1}", gridlines=gridlines)

    # Plotting test data
    for t in range(n_test_pairs):
        col_idx = n_train_pairs + t
        plot_grid(axs[0, col_idx], test_input_grids[t], title=f"Test Input {t + 1}", gridlines=gridlines)
        plot_grid(axs[1, col_idx], test_output_grids[t], title=f"Test Output {t + 1}", gridlines=gridlines)

    plt.tight_layout()

    if return_base64:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png", bbox_inches="tight")
        img_buf.seek(0)
        base64_image = base64.b64encode(img_buf.read()).decode("utf-8")
        plt.close(fig)
        return base64_image
    else:
        plt.show()
        return fig  # Return the Figure object for use in notebooks


# --- Main Visualization Entry Points ---


def visualize_task_pair_from_json(task_data, example_type="train", example_index=0, title_prefix="", gridlines=True):
    """
    Visualizes a single input-output example from a loaded ARC task dictionary.
    This function wraps `plot_riddle_two_rows` to display a single pair.

    Args:
        task_data (dict): The loaded ARC task dictionary.
        example_type (str): Specifies whether to visualize from 'train' or 'test' examples.
        example_index (int): The index of the specific example within the chosen set.
        title_prefix (str): A string to prepend to the overall plot title.
        gridlines (bool): If True, draws grid lines on the plots.

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object, or None if an error occurs.
    """
    if example_type not in task_data:
        print(f"Error: Task data does not contain '{example_type}' examples.")
        return None

    examples = task_data[example_type]

    if not (0 <= example_index < len(examples)):
        print(
            "Error: example_index "
            f"{example_index} is out of bounds for the {example_type} data (0 to {len(examples) - 1})."
        )
        return None

    input_grid_data = examples[example_index]["input"]
    output_grid_data = examples[example_index].get("output")  # Use .get to safely handle missing output

    input_grid = list_to_grid(input_grid_data)

    # Handle missing output grid for test cases gracefully
    output_grid = (
        list_to_grid(output_grid_data)
        if output_grid_data is not None
        else np.full_like(input_grid, 0)
    )

    # Use plot_riddle_two_rows to display this single pair.
    # Pass empty lists for the other sets as we only want to show one pair.
    if example_type == "train":
        fig = plot_riddle_two_rows([input_grid], [output_grid], [], [], gridlines=gridlines)
    else:  # example_type == "test"
        fig = plot_riddle_two_rows([], [], [input_grid], [output_grid], gridlines=gridlines)

    if fig:
        # Add a suptitle to the figure
        fig.suptitle(f"{title_prefix}{example_type.capitalize()} Example {example_index + 1}", y=1.02)
        return fig
    return None


def visualize_nlp_riddle_two_rows(nlp_string, grid_starting_row=4, title="", gridlines=True, max_train_pairs=8):
    """
    Visualizes an ARC riddle from its NLP string representation in a two-row layout.
    It first converts the NLP string to ARC arrays, then plots them.

    Args:
        nlp_string (str): The NLP string representation of an ARC task.
        grid_starting_row (int): A parameter for the NLP string parsing, indicating where grid data starts.
        title (str): The title for the plot.
        gridlines (bool): If True, draws grid lines on the plots.
        max_train_pairs (int): Maximum number of training pairs to display (default: 8).

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object, or None if an error occurs.
    """
    try:
        train_inputs, train_outputs, test_inputs, test_outputs = convert_nlp_arc_string_to_arc_arrays(
            nlp_string, grid_starting_row
        )
        fig = plot_riddle_two_rows(
            train_inputs, train_outputs, test_inputs, test_outputs, gridlines=gridlines, max_train_pairs=max_train_pairs
        )
        if fig:
            fig.suptitle(title, y=1.02)
        return fig
    except Exception as e:
        print(f"Error visualizing NLP riddle: {e}")
        return None

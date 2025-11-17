"""
Grid utilities module for ARC Prize 2025 solution
Contains grid validation, conversion, and transformation functions
"""

from collections import Counter
import re
from typing import Any


try:
    from .config import MAX_GRID_SIZE, MAX_SYMBOLS, SYMBOL_ENCODING
except ImportError:
    # Fallback for when script is run directly (e.g., in Kaggle)
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import MAX_GRID_SIZE, MAX_SYMBOLS  # type: ignore

    SYMBOL_ENCODING = {"enabled": False, "scheme": "letters"}  # type: ignore

try:
    from . import config as _config_module
except ImportError:
    _config_module = None


def _is_debug_enabled() -> bool:
    """Helper that checks runtime debug flags without importing logging globally."""
    if _config_module is None:
        return False
    try:
        return bool(getattr(_config_module, "DEBUG_MODE", False) or getattr(_config_module, "VERBOSE_LOGGING", False))
    except Exception:
        return False


def _debug_print(message: str) -> None:
    if _is_debug_enabled():
        print(message)


# ------------------------------------------------------------
# Symbol encoding helpers (digits <-> letters)
# ------------------------------------------------------------


_DIGITS = "0123456789"
_LETTERS = "abcdefghij"


def _encode_symbol(val: int) -> str:
    if (
        SYMBOL_ENCODING.get("enabled")
        and SYMBOL_ENCODING.get("scheme") == "letters"
        and isinstance(val, int)
        and 0 <= val < 10
    ):
        return _LETTERS[val]
    return str(val)


def decode_symbols_in_text(text: str) -> str:
    """Map letter symbols back to digits in free-form text.
    a->0, b->1, ..., j->9 (case-insensitive). Other chars preserved.
    """
    if not isinstance(text, str):
        text = str(text)
    # Build translation table for both lowercase and uppercase letters
    trans = {}
    for i, ch in enumerate(_LETTERS):
        trans[ord(ch)] = ord(_DIGITS[i])
        trans[ord(ch.upper())] = ord(_DIGITS[i])
    return text.translate(trans)


def shape(grid: list[list[int]]) -> tuple[int, int]:
    """Returns (height, width) of a grid."""
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        return (0, 0)
    return (len(grid), len(grid[0]))


def is_grid(grid: Any) -> bool:
    """Validates if input is a valid ARC grid."""
    try:
        if not isinstance(grid, list) or not grid:
            return False
        if not (0 < len(grid) <= MAX_GRID_SIZE):
            return False
        if not all(isinstance(row, list) for row in grid):
            return False
        first_row_len = len(grid[0])
        if not (0 < first_row_len <= MAX_GRID_SIZE):
            return False
        if not all(len(row) == first_row_len for row in grid):
            return False
        return all(
            isinstance(x, int) and 0 <= x < MAX_SYMBOLS
            for row in grid
            for x in row
        )
    except Exception:
        return False


def is_valid_prediction(grid: Any) -> bool:
    """Validates if a grid is a valid prediction (not None and not fallback patterns)."""
    if grid is None:
        return False
    if not is_grid(grid):
        return False
    # Exclude common fallback patterns that indicate parsing failures
    if grid == [[0]]:  # Single zero cell fallback
        return False
    return not (len(grid) == 1 and len(grid[0]) == 1 and grid[0][0] == 0)


def grid_to_string(grid: list[list[int]]) -> str:
    """Converts grid to string representation."""
    if not is_grid(grid):
        return ""
    try:
        # Encode symbols (optionally to letters) row-wise without spaces between cells
        return " ".join(["".join(_encode_symbol(cell) for cell in row) for row in grid])
    except Exception:
        return ""


def _grid_rows_to_strings(grid: list[list[int]]) -> list[str]:
    """Return each row as a compact string (without spaces between cells)."""
    if not is_grid(grid):
        return []
    try:
        return ["".join(_encode_symbol(cell) for cell in row) for row in grid]
    except Exception:
        return []


def format_grid_display(grid: list[list[int]]) -> str:
    """Formats grid for human-readable display with newlines and visual separation."""
    if not is_grid(grid):
        return "INVALID_GRID"
    try:
        # Create visual grid with spaces between digits and newlines between rows
        formatted_rows = []
        for row in grid:
            formatted_row = " ".join(str(cell) for cell in row)
            formatted_rows.append(formatted_row)
        return "\n".join(formatted_rows)
    except Exception as e:
        return f"GRID_FORMAT_ERROR: {e}"


def output_prefix(grid):
    outputnums = []
    for r in grid:
        for sym in r:
            if sym not in outputnums:
                outputnums.append(sym)
    # Encode unique symbols as a compact string (optionally letters)
    outputnums = "".join([_encode_symbol(sym) for sym in outputnums])
    h, w = shape(grid)
    totsiz = w * h + h - 1
    return f"{totsiz} {h} {w} {outputnums} "


def makeprompt(task: dict, style: str = "legacy") -> str:
    """Generates prompt string for a task."""
    if str(style or "").lower() == "arc_diffusion":
        try:
            train_examples = task.get("train", [])
            if not isinstance(train_examples, list):
                train_examples = []
            valid_examples = [
                ex
                for ex in train_examples
                if isinstance(ex, dict) and is_grid(ex.get("input")) and is_grid(ex.get("output"))
            ]
            if not valid_examples:
                return "[no_valid_train_examples]"

            lines: list[str] = ["# Train Examples"]
            for idx, example in enumerate(valid_examples, start=1):
                lines.append(f"## Example {idx}")
                lines.append("Input:")
                lines.extend(_grid_rows_to_strings(example["input"]))
                lines.append("")
                lines.append("Output:")
                lines.extend(_grid_rows_to_strings(example["output"]))
                lines.append("")

            test_examples = task.get("test", [])
            if not (isinstance(test_examples, list) and test_examples):
                return "[no_test_data]"
            test_example = test_examples[0]
            if not isinstance(test_example, dict) or not is_grid(test_example.get("input")):
                return "[missing_input]"

            lines.append("# Test Input")
            lines.extend(_grid_rows_to_strings(test_example["input"]))

            # Remove potential trailing blank lines while preserving intentional spacing
            while lines and lines[-1] == "":
                lines.pop()
            return "\n".join(lines).strip()
        except Exception:
            return "[prompt_error]"

    try:
        prompt = "solve: train"
        train_examples = task.get("train", [])
        if not isinstance(train_examples, list):
            train_examples = []

        valid_count = 0
        for idx, ex in enumerate(train_examples):
            if not isinstance(ex, dict) or "input" not in ex or "output" not in ex:
                continue
            input_str = grid_to_string(ex["input"])
            output_str = grid_to_string(ex["output"])
            if not input_str or not output_str:
                continue
            valid_count += 1
            prompt += f" input{idx + 1} {input_str}"
            prompt += f" output{idx + 1} {output_prefix(ex['output'])}{output_str}."

        if valid_count == 0:
            prompt += " [no_valid_train_examples]"

        prompt += " test"
        test_examples = task.get("test", [])
        if isinstance(test_examples, list) and test_examples:
            test_ex = test_examples[0]
            if isinstance(test_ex, dict) and "input" in test_ex:
                test_input_str = grid_to_string(test_ex["input"])
                if test_input_str:
                    prompt += f" tinput1 {test_input_str} toutput1 "
                else:
                    prompt += " tinput1 [invalid_grid] toutput1 "
            else:
                prompt += " tinput1 [missing_input] toutput1 "
        else:
            prompt += " tinput1 [no_test_data] toutput1 "

        return prompt
    except Exception:
        return "solve: train [prompt_error] test tinput1 [prompt_error] toutput1 "


def output_to_string(output: str) -> str:
    """Extracts grid string from model output."""
    # First decode any letter symbols back to digits for robust parsing
    output = decode_symbols_in_text(output)

    # Normalize newlines for downstream parsing
    normalized_output = str(output).replace("\r\n", "\n").replace("\r", "\n")

    # remove anything non-numeric, non-literal-space, newline, or non-period
    cleaned_output = re.sub(r"[^0-9 .\n]", "", normalized_output)
    # take the part of the string before the period
    cleaned_for_primary = cleaned_output.replace("\\n", " ").replace("\n", " ")
    output_parts = cleaned_for_primary.strip().split(".")[0].strip().split(" ")
    valid_parts = [part for part in output_parts if part]

    # Skip first 4 parts (size, height, width, symbols)
    grid_parts = valid_parts[4:] if len(valid_parts) > 4 else []
    result = " ".join(grid_parts)
    if result:
        return result

    # Fallback: parse newline-delimited digit rows (Dream diffusion style)
    digit_lines: list[str] = []
    for line in normalized_output.splitlines():
        digits = re.findall(r"[0-9]", line)
        if digits:
            digit_lines.append("".join(digits))
    return " ".join(digit_lines)


def output_to_grid(output: str) -> list[list[int]] | None:
    """Converts model output to grid. Returns None for invalid parsing instead of [[0]]."""
    try:
        grid_str = output_to_string(output)
        if not grid_str:
            return None

        parsed_rows_intermediate = []
        # Split grid_str into potential row strings and parse each
        for row_str in grid_str.split(" "):
            if not row_str:  # Skip empty strings that can result from multiple spaces
                continue
            try:
                # Convert character digits in row_str to integers
                row = [int(c) for c in row_str]
                parsed_rows_intermediate.append(row)
            except ValueError as ve:
                # Handle cases where a row_str contains non-digits
                _debug_print(f"[DEBUG] ValueError parsing row string '{row_str}': {ve}")
                return None

        if not parsed_rows_intermediate:
            # This means grid_str was non-empty but contained no valid row_str parts
            _debug_print(f"[DEBUG] No valid rows parsed from grid_str: '{grid_str}'")
            return None

        # Determine the maximum length among all successfully parsed rows
        max_len = 0
        for r_val in parsed_rows_intermediate:
            if len(r_val) > max_len:
                max_len = len(r_val)

        # If max_len is 0 (e.g., all parsed rows were empty lists),
        # is_grid will later invalidate it, as ARC grids must have width > 0.

        final_rows = []
        for p_row in parsed_rows_intermediate:
            current_len = len(p_row)

            if current_len == max_len:
                final_rows.append(p_row)
            elif current_len < max_len:
                # This row is shorter than the longest row and needs padding.
                if not p_row:
                    # An empty row cannot be padded using its "last character".
                    _debug_print(
                        "[DEBUG] Cannot pad an initially empty parsed row to meet max_len "
                        f"{max_len}. Original grid_str: '{grid_str}'"
                    )
                    return None

                last_char = p_row[-1]  # Get the character to use for padding
                # Pad the row by appending the last_char
                padded_row = p_row + [last_char] * (max_len - current_len)
                final_rows.append(padded_row)
                # print(f"[DEBUG] Padded a row from length {current_len} to {max_len} using char '{last_char}'.")
            else:  # current_len > max_len
                # This case should not be reached if max_len was correctly found.
                _debug_print(
                    f"[DEBUG] Row length {current_len} is unexpectedly greater than calculated "
                    f"max_len {max_len}. Row: {p_row}. Original grid_str: '{grid_str}'"
                )
                return None

        # Validate the fully processed grid.
        if not is_grid(final_rows):
            h_debug = len(final_rows)
            w_debug = len(final_rows[0]) if h_debug > 0 and final_rows[0] is not None else 0
            sample_debug_str = (
                "".join(map(str, final_rows[0][:5]))
                if h_debug > 0 and final_rows[0]
                else ("[]" if h_debug > 0 else "N/A")
            )
            _debug_print(
                "[DEBUG] Final grid is invalid according to is_grid. "
                f"Shape after processing: {h_debug}x{w_debug}, sample of first row: '{sample_debug_str}'. "
                f"Original grid_str: '{grid_str}'"
            )
            return None

        return final_rows

    except Exception as e:
        # Catch-all for any other unexpected errors.
        _debug_print(f"[DEBUG] Exception in output_to_grid: {e}. Output was: {output[:200]}...")
        return None


def grid_string_to_grid(grid_str: str) -> list[list[int]] | None:
    """Parse a compact grid string like '123 456' into a grid without extra validation noise."""
    if not isinstance(grid_str, str) or not grid_str.strip():
        return None
    try:
        rows = [[int(ch) for ch in row] for row in grid_str.strip().split(" ") if row]
        return rows if is_grid(rows) else None
    except Exception:
        return None


def prompt_to_arc_task(prompt: str, correct_answer: str | None = None) -> dict | None:
    """Best-effort reconstruction of an ARC task dict from a serialized prompt and optional answer."""
    if not isinstance(prompt, str) or not prompt.startswith("solve:"):
        return None
    try:
        cleaned = re.sub(r"([.,])", r" \1 ", prompt)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        m_test = re.search(r"\btest\b", cleaned)
        if not m_test:
            return None

        train_section = cleaned[: m_test.start()].replace("solve: train", "").strip()
        test_section = cleaned[m_test.end() :].strip()

        train_examples = []
        idx = 1
        while True:
            match_in = re.search(rf"\binput{idx}\b (.*?) \boutput{idx}\b", train_section)
            if not match_in:
                break
            tail = train_section[match_in.end() :]
            match_next = re.search(rf"\binput{idx + 1}\b| \.[ ]?| \.$", tail)
            out_segment = tail[: match_next.start()] if match_next else tail
            g_input = grid_string_to_grid(match_in.group(1))
            g_output = output_to_grid(out_segment)
            if not (g_input and g_output):
                break
            train_examples.append({"input": g_input, "output": g_output})
            idx += 1

        if not train_examples:
            return None

        match_test = re.search(r"\btinput1\b (.*?) \btoutput1\b", test_section)
        if not match_test:
            return None
        test_input = grid_string_to_grid(match_test.group(1))
        if test_input is None:
            return None

        entry: dict[str, Any] = {"input": test_input}

        expect_output = correct_answer is not None and str(correct_answer).strip() != ""
        if expect_output:
            test_output = grid_string_to_grid(correct_answer) or output_to_grid(correct_answer)
            if test_output is None or not is_grid(test_output):
                return None
            entry["output"] = test_output

        return {
            "train": train_examples,
            "test": [entry],
        }
    except Exception:
        return None


def scramble_grid_cells(grid: list[list[int]], frac: float, num_syms: int = 10) -> list[list[int]]:
    """Replace a fraction of grid cells with random symbols."""
    if not is_grid(grid):
        return grid
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is expected but handle gracefully
        np = None

    h = len(grid)
    w = len(grid[0]) if h else 0
    total = h * w
    count = max(1, round(max(0.0, min(1.0, float(frac))) * total)) if total else 0
    if count <= 0:
        return [row[:] for row in grid]
    if np is not None:
        flat_indices = np.random.choice(total, size=min(count, total), replace=False)

        def rand_symbol() -> int:
            return int(np.random.randint(0, num_syms))
    else:
        import random as _rnd

        flat_indices = _rnd.sample(range(total), k=min(count, total))

        def rand_symbol() -> int:
            return _rnd.randrange(num_syms)
    corrupted = [row[:] for row in grid]
    for idx in flat_indices:
        r = idx // w
        c = idx % w
        corrupted[r][c] = rand_symbol()
    return corrupted


def resize_grid(input_grid: list[list[int]], target_rows: int, target_cols: int) -> list[list[int]]:
    """Resize grid with nearest-neighbor or majority pooling semantics."""
    if not is_grid(input_grid) or target_rows <= 0 or target_cols <= 0:
        return [[0 for _ in range(max(1, target_cols))] for _ in range(max(1, target_rows))]

    in_rows = len(input_grid)
    in_cols = len(input_grid[0]) if in_rows else 0
    down_r = target_rows < in_rows
    down_c = target_cols < in_cols
    resized = []
    for r in range(target_rows):
        row_out = []
        for c in range(target_cols):
            if down_r:
                r0 = int(r * in_rows / target_rows)
                r1 = int((r + 1) * in_rows / target_rows)
            else:
                frac_r = r * (in_rows - 1) / (target_rows - 1) if target_rows > 1 else 0
                r0 = r1 = min(round(frac_r), in_rows - 1)
            if down_c:
                c0 = int(c * in_cols / target_cols)
                c1 = int((c + 1) * in_cols / target_cols)
            else:
                frac_c = c * (in_cols - 1) / (target_cols - 1) if target_cols > 1 else 0
                c0 = c1 = min(round(frac_c), in_cols - 1)

            if down_r or down_c:
                patch = [input_grid[i][j] for i in range(r0, max(r1, r0 + 1)) for j in range(c0, max(c1, c0 + 1))]
                if patch:
                    counts = Counter(patch)
                    selected = max(counts.items(), key=lambda x: (x[1], x[0]))[0]
                else:
                    selected = input_grid[r0][c0]
            else:
                selected = input_grid[r0][c0]
            row_out.append(int(selected))
        resized.append(row_out)
    return resized


def text_to_grid_string(text: str) -> str | None:
    """Convert arbitrary model text into a normalized grid string when possible."""
    grid = output_to_grid(text)
    if not grid:
        grid = grid_string_to_grid(text)
    if not grid:
        cleaned = re.sub(r"[^0-9 ]+", " ", text or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        grid = grid_string_to_grid(cleaned)
    return grid_to_string(grid) if grid and is_grid(grid) else None

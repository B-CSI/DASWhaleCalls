# %% Libraries
import re
import pandas as pd

# %% read_csv_metadata function:
def read_csv_metadata(csv_path):
    """
    Read metadata stored as commented lines (# ...) at the
    beginning of the KVP CSV file.

    Returns
    -------
    metadata : dict
        Dictionary containing the metadata parameters.
    """

    metadata = {}

    with open(
        csv_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            # Stop when the actual CSV data begins
            if not line.startswith("#"):
                break

            # Remove '#' and surrounding whitespace
            line = line[1:].strip()

            # Ignore empty/comment-only lines
            if not line:
                continue

            # Only process key = value entries
            if "=" not in line:
                continue

            key, value = line.split(
                "=",
                1
            )

            key = key.strip()
            value = value.strip()

            # ------------------------------------------------
            # Try to convert numerical values automatically
            # ------------------------------------------------

            try:

                # Integer
                if re.fullmatch(
                    r"[+-]?\d+",
                    value
                ):
                    value = int(value)

                # Float
                elif re.fullmatch(
                    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
                    value
                ):
                    value = float(value)

            except ValueError:
                pass

            metadata[key] = value

    return metadata
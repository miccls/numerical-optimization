import gzip
import io
import ssl
import urllib.request
from typing import Any

import numpy as np
from common.numpy_type_aliases import ArrayF


def download_and_parse_mps(
    problem_name: str,
) -> (
    tuple[
        ArrayF,
        ArrayF,
        ArrayF,
        list[str],
        dict[str, int],
        dict[str, int],
    ]
    | None
):
    url = f"https://raw.githubusercontent.com/coin-or-tools/Data-Netlib/master/{problem_name}.mps.gz"
    print(f"Downloading {problem_name} from {url}...")

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req, context=ctx)

        compressed_file = io.BytesIO(response.read())
        mps_text: str = gzip.GzipFile(fileobj=compressed_file).read().decode("utf-8")

    except Exception as e:
        print(f"Failed to download: {e}")
        return None

    row_map: dict[str, int] = {}
    col_map: dict[str, int] = {}
    row_types: list[str] = []
    a_dict: dict[tuple[int, int], float] = {}
    b_dict: dict[int, float] = {}
    c_dict: dict[int, float] = {}
    bounds: dict[int, dict[str, Any]] = {}  # Dictionary to hold bound data
    obj_name: str | None = None
    section: str | None = None

    for line in mps_text.splitlines():
        if line.startswith("ROWS"):
            section = "ROWS"
            continue
        elif line.startswith("COLUMNS"):
            section = "COLUMNS"
            continue
        elif line.startswith("RHS"):
            section = "RHS"
            continue
        elif line.startswith("BOUNDS"):
            section = "BOUNDS"
            continue
        elif line.startswith("ENDATA"):
            break

        parts = line.split()
        if not parts:
            continue

        if section == "ROWS":
            rtype, rname = parts[0], parts[1]
            if rtype == "N" and obj_name is None:
                obj_name = rname
            elif rtype != "N" and rname not in row_map:
                row_map[rname] = len(row_map)
                row_types.append(rtype)

        elif section == "COLUMNS":
            cname = parts[0]
            if cname not in col_map:
                col_map[cname] = len(col_map)
            c_idx = col_map[cname]

            rname1, val1 = parts[1], float(parts[2])
            if rname1 == obj_name:
                c_dict[c_idx] = val1
            elif rname1 in row_map:
                a_dict[(row_map[rname1], c_idx)] = val1

            if len(parts) >= 5:
                rname2, val2 = parts[3], float(parts[4])
                if rname2 == obj_name:
                    c_dict[c_idx] = val2
                elif rname2 in row_map:
                    a_dict[(row_map[rname2], c_idx)] = val2

        elif section == "RHS":
            rname1, val1 = parts[1], float(parts[2])
            if rname1 in row_map:
                b_dict[row_map[rname1]] = val1
            if len(parts) >= 5:
                rname2, val2 = parts[3], float(parts[4])
                if rname2 in row_map:
                    b_dict[row_map[rname2]] = val2

        # --- Parse BOUNDS ---
        elif section == "BOUNDS":
            # Format: Type BoundName ColName Value
            b_type = parts[0]
            bound_cname: str | None = parts[2] if len(parts) > 2 else None

            if bound_cname is not None and bound_cname in col_map:
                c_idx = col_map[bound_cname]
                if c_idx not in bounds:
                    bounds[c_idx] = {}

                if b_type in ["LO", "UP", "FX"]:
                    bounds[c_idx][b_type] = float(parts[3])
                elif b_type in ["FR", "MI"]:
                    bounds[c_idx][b_type] = True

    # --- Construct Matrices (Including extra rows for bounds) ---
    extra_rows = 0
    for bnd in bounds.values():
        if "UP" in bnd:
            extra_rows += 1
        if "LO" in bnd and bnd["LO"] != 0.0:
            extra_rows += 1
        if "FX" in bnd:
            extra_rows += 1

    m, n = len(row_map) + extra_rows, len(col_map)
    a: ArrayF = np.zeros((m, n))
    b: ArrayF = np.zeros(m)
    c: ArrayF = np.zeros(n)

    for (i, j), val in a_dict.items():
        a[i, j] = val
    for i, val in b_dict.items():
        b[i] = val
    for j, val in c_dict.items():
        c[j] = val

    # --- Append Bounds as Matrix Constraints ---
    current_row = len(row_map)
    for c_idx, bnd in bounds.items():
        if "UP" in bnd:
            a[current_row, c_idx] = 1.0
            b[current_row] = bnd["UP"]
            row_types.append("L")  # Less than or equal
            current_row += 1
        if "LO" in bnd and bnd["LO"] != 0.0:
            a[current_row, c_idx] = 1.0
            b[current_row] = bnd["LO"]
            row_types.append("G")  # Greater than or equal
            current_row += 1
        if "FX" in bnd:
            a[current_row, c_idx] = 1.0
            b[current_row] = bnd["FX"]
            row_types.append("E")  # Exact equality
            current_row += 1

        # A warning for variables that are allowed to be negative
        if "FR" in bnd or "MI" in bnd or ("LO" in bnd and bnd["LO"] < 0):
            print(
                f"WARNING: Column {c_idx} allows negative values. Standard form conversion requires variable splitting."
            )

    print(
        f"Parsed successfully: Base a is {len(row_map)}x{n}, plus {extra_rows} bound rows."
    )
    return a, b, c, row_types, col_map, row_map


def convert_to_standard_form(
    a: ArrayF, b: ArrayF, c: ArrayF, row_types: list[str]
) -> tuple[ArrayF, ArrayF, ArrayF]:
    """
    Converts mixed inequalities into standard form (Ax = b)
    by appending slack and surplus variables.
    """
    m, n = a.shape

    # Count how many slack/surplus variables we need ('L' or 'G')
    num_slacks = sum(1 for rtype in row_types if rtype in ["L", "G"])

    # Allocate new expanded matrices
    a_std: ArrayF = np.zeros((m, n + num_slacks))
    c_std: ArrayF = np.zeros(n + num_slacks)

    # Copy original data into the left side of the new matrices
    a_std[:, :n] = a
    c_std[:n] = c

    # Populate the diagonal of the slack portion
    slack_idx = n
    for i, rtype in enumerate(row_types):
        if rtype == "L":
            # <= constraint gets a positive slack variable
            a_std[i, slack_idx] = 1.0
            slack_idx += 1
        elif rtype == "G":
            # >= constraint gets a negative surplus variable
            a_std[i, slack_idx] = -1.0
            slack_idx += 1

    # 'E' (equality) constraints are skipped because they don't need slacks

    print(f"Added {num_slacks} slack/surplus variables.")
    print(f"New a shape: {a_std.shape}, New c shape: {c_std.shape}")

    return a_std, b, c_std


# --- Test the parser ---
if __name__ == "__main__":
    # 'afiro' is the smallest netlib problem (27 constraints, 32 variables)
    result = download_and_parse_mps("afiro")

    if result:
        a_res, b_res, c_res, row_types_res, col_map_res, row_map_res = result
        print("\nConstraint Matrix (a) shape:", a_res.shape)
        print("Right-Hand Side (b) shape:", b_res.shape)
        print("Objective Vector (c) shape:", c_res.shape)

        # Sneak peek at the first 5 objective values
        print("\nFirst 5 objective coefficients:", c_res[:5])

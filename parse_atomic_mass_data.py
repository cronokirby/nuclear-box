import csv
import os

ABSENT_MARKER = "absent"
CALCULATED_MARKER = "calculated"
ESTIMATED_MARKER = "estimated"
ABSENT_CHARACTER = "*"
ESTIMATED_CHARACTER = "#"


def is_parseable_as_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def parse_quantity(row, i):
    if row[i] == ABSENT_CHARACTER:
        return 1, ["", "", ABSENT_MARKER]
    estimated = ESTIMATED_CHARACTER in row[i]
    marker = ESTIMATED_MARKER if estimated else CALCULATED_MARKER
    return 2, [
        float(row[i].replace(ESTIMATED_CHARACTER, "")),
        float(row[i + 1].replace(ESTIMATED_CHARACTER, "")),
        marker,
    ]


def convert_row(line):
    # First, element the leading Fortran control character
    line = line[1:]
    parts = line.split()
    out = [
        # parts[0] is N - Z, which is stupid to include imo
        # parts[1] is N
        int(parts[1]),
        # parts[2] is Z
        int(parts[2]),
        # parts[3] is A
        int(parts[3]),
        # parts[4] is the element name, as a string
        parts[4],
    ]
    index = 5
    # We may have to skip a column if there's a decay mode here
    if not is_parseable_as_float(parts[index]):
        index += 1
    # Now we have the excess mass (energy equivalent)
    adjust, next_part = parse_quantity(parts, index)
    out += next_part
    index += adjust
    # Then we have the binding energy
    adjust, next_part = parse_quantity(parts, index)
    out += next_part
    index += adjust
    # Skip over the column containing 'B-'
    index += 1
    # Now we have the beta decay energy
    adjust, next_part = parse_quantity(parts, index)
    out += next_part
    index += adjust
    # Finally, atomic mass
    _, next_part = parse_quantity(parts, index + 1)
    out += [float(parts[index]) + next_part[0] * 1e-6] + next_part[1:]

    return out


def convert(in_file, out_file):
    with open(out_file, "w") as out_fp:
        writer = csv.writer(out_fp)
        writer.writerow(
            [
                "N",
                "Z",
                "A",
                "Element",
                "Mass Excess (keV)",
                "ME Error Margin",
                "ME Calculated?",
                "Binding Energy Per Nucleon (keV)",
                "BEN Error Margin",
                "BEN Calculated?",
                "Beta Decay Energy (keV)",
                "BDE Error Margin",
                "BDE Calculated?",
                "Atomic Mass (u)",
                "AM Error Margin",
                "AM Calculated?",
            ]
        )
        with open(in_file, "r") as in_fp:
            for line in in_fp:
                writer.writerow(convert_row(line))


if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    convert("data/atomic_mass_data.txt", "data/atomic_mass_data.csv")

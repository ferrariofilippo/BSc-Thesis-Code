"""Extract training losses from numbered run folders into a CSV file."""

import csv
import os
import re


OUTPUT_CSV = "losses.csv"

# Accept decimal values as well as values written in scientific notation.
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

EPOCH_LINE_RE = re.compile(
    rf"epoch=\d+\s+train_loss=({NUMBER})\s+val_loss=({NUMBER})"
)
BEST_VAL_LOSS_RE = re.compile(rf"Best\s+val_loss=({NUMBER})", re.IGNORECASE)

# Match the run number in names such as "122_model.hidden_1_dim=8,...".
FOLDER_NUMBER_RE = re.compile(r"^(\d+)(?:_|$)")


def parse_train_log(filepath):
    """Return (best_val_loss, train_loss, val_loss) from a train.log file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as log_file:
        contents = log_file.read()

    epoch_matches = list(EPOCH_LINE_RE.finditer(contents))
    best_matches = list(BEST_VAL_LOSS_RE.finditer(contents))

    if not epoch_matches:
        return None, None, None

    last_epoch = epoch_matches[-1]
    train_loss = float(last_epoch.group(1))
    val_loss = float(last_epoch.group(2))
    best_val_loss = float(best_matches[-1].group(1)) if best_matches else None

    return best_val_loss, train_loss, val_loss


def main():
    cwd = os.getcwd()
    output_path = os.path.join(cwd, OUTPUT_CSV)
    rows = []

    folders = []
    for entry in os.listdir(cwd):
        folder_path = os.path.join(cwd, entry)
        folder_match = FOLDER_NUMBER_RE.match(entry)

        if os.path.isdir(folder_path) and folder_match:
            folders.append((int(folder_match.group(1)), entry, folder_path))

    for folder_id, folder_name, folder_path in sorted(folders):
        log_path = os.path.join(folder_path, "train.log")

        if not os.path.isfile(log_path):
            print(f"[WARN] No train.log in folder '{folder_name}', skipping.")
            continue

        best_val_loss, train_loss, val_loss = parse_train_log(log_path)

        if train_loss is None or val_loss is None:
            print(f"[WARN] No epoch loss values found in '{log_path}', skipping.")
            continue

        if best_val_loss is None:
            print(f"[WARN] No best val_loss found in '{log_path}'; leaving it blank.")

        rows.append((folder_id, best_val_loss, train_loss, val_loss))

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["FOLDER_ID", "BEST_VAL_LOSS", "TRAIN_LOSS", "VAL_LOSS"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to '{output_path}'.")


if __name__ == "__main__":
    main()
    
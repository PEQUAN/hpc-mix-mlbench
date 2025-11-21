import json
import csv
import os
import re

CATEGORY_DISPLAY_NAMES = {
    'double': 'FP64',
    'float': 'FP32',
    'half_float::half': 'FP16',
    'flx::floatx<10, 5>': 'FP16',
    'flx::floatx<8, 7>': 'BF16',
    'flx::floatx<4, 3>': 'E4M3',
    'flx::floatx<5, 2>': 'E5M2'
}

ROOT_DIR = '.'  # root folder

# List to store data for the summary CSV
summary_data = []

# Regex to detect arbitrary JSON input files
JSON_PATTERN = re.compile(r"prec_setting_([0-9]+)\.json$")

# Get list of folders in ROOT_DIR
folders = [
    f for f in os.listdir(ROOT_DIR)
    if os.path.isdir(os.path.join(ROOT_DIR, f))
]

for folder in folders:
    if folder == "to-do":
        continue

    folder_path = os.path.join(ROOT_DIR, folder)

    # Detect all matching JSON files in this folder
    json_files = []
    for file in os.listdir(folder_path):
        match = JSON_PATTERN.match(file)
        if match:
            index = int(match.group(1))
            json_files.append((index, file))

    # No matching files → skip folder
    if not json_files:
        print(f"No prec_setting_<i>.json files found in {folder_path}, skipping folder.")
        continue

    # Sort by extracted index (prec_setting_i.json where i increases)
    json_files.sort(key=lambda x: x[0])

    table_data = []

    for idx, json_file in json_files:
        file_path = os.path.join(folder_path, json_file)

        # Read JSON
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Warning: Failed to read {file_path}, skipping.")
            continue

        # Process data entries
        for dict_idx, entry in enumerate(data, 1):
            row = [idx, dict_idx]   # Precision Setting, Significant Digits
            summary_row = [folder, idx, dict_idx]

            for key, display_name in CATEGORY_DISPLAY_NAMES.items():
                count = len(entry.get(key, []))
                row.append(count)
                summary_row.append(count)

            table_data.append(row)
            summary_data.append(summary_row)

    # Write per-folder CSV
    if table_data:
        headers = ['Precision Setting', 'Significant Digits'] + \
                  [CATEGORY_DISPLAY_NAMES[key] for key in CATEGORY_DISPLAY_NAMES]

        output_csv = os.path.join(folder_path, 'fp_counts_all.csv')
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(table_data)
        print(f"Generated CSV: {output_csv}")
    else:
        print(f"No valid JSON data in {folder_path}, skipping CSV creation.")

# Write global summary CSV
if summary_data:
    summary_headers = ['Folder', 'Precision Setting', 'Significant Digits'] + \
                      [CATEGORY_DISPLAY_NAMES[key] for key in CATEGORY_DISPLAY_NAMES]

    summary_csv = os.path.join(ROOT_DIR, 'fp_counts_summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(summary_headers)
        writer.writerows(summary_data)

    print(f"Generated Summary CSV: {summary_csv}")
else:
    print("No valid data found across folders, skipping summary CSV.")

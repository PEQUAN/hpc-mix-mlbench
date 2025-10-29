import json
import csv
import os

CATEGORY_DISPLAY_NAMES = {
    'double': 'FP64',
    'float': 'FP32',
    'half_float::half': 'FP16',
    'flx::floatx<8, 7>': 'BF16',
    'flx::floatx<4, 3>': 'E4M3',
    'flx::floatx<5, 2>': 'E5M2'
}

JSON_FILES = [
    'precision_settings_1.json',
    'precision_settings_2.json',
    'precision_settings_3.json',
    'precision_settings_4.json'
]

# Root directory to search for folders (use current directory if empty)
ROOT_DIR = '.'  # Change to specific path if needed, e.g., '/path/to/folders'

# List to store data for the summary CSV
summary_data = []

# Get list of folders in ROOT_DIR
folders = [f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))]

for folder in folders:
    if folder == "to-do":
        continue
    folder_path = os.path.join(ROOT_DIR, folder)
    
    # Check if all required JSON files exist in the folder
    all_files_present = True
    for json_file in JSON_FILES:
        if not os.path.isfile(os.path.join(folder_path, json_file)):
            print(f"Warning: File {json_file} not found in {folder_path}, skipping folder.")
            all_files_present = False
            break
    
    if not all_files_present:
        continue
    
    # Initialize table_data for per-folder CSV
    table_data = []
    
    for file_idx, json_file in enumerate(JSON_FILES, 1):
        file_path = os.path.join(folder_path, json_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found, skipping.")
            continue
        except json.JSONDecodeError:
            print(f"Warning: File {file_path} is not a valid JSON, skipping.")
            continue

        # Process each dictionary in the file
        for dict_idx, entry in enumerate(data, 1):
            row = [file_idx, dict_idx]  # Columns for Precision Setting and Significant Digits
            summary_row = [folder, file_idx, dict_idx]  # Columns for summary: Folder, Precision Setting, Significant Digits
            for key, display_name in CATEGORY_DISPLAY_NAMES.items():
                count = len(entry.get(key, []))  # Get length of list for key, default to 0 if missing
                row.append(count)
                summary_row.append(count)
            table_data.append(row)
            summary_data.append(summary_row)  # Add to summary data

    # Generate per-folder CSV
    if table_data:
        headers = ['Precision Setting', 'Significant Digits'] + [CATEGORY_DISPLAY_NAMES[key] for key in CATEGORY_DISPLAY_NAMES]
        output_csv = os.path.join(folder_path, 'fp_counts_all.csv')
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(table_data)
        print(f"Generated CSV: {output_csv}")
    else:
        print(f"No valid data found in {folder_path}, skipping CSV creation.")

# Generate summary CSV
if summary_data:
    summary_headers = ['Folder', 'Precision Setting', 'Significant Digits'] + [CATEGORY_DISPLAY_NAMES[key] for key in CATEGORY_DISPLAY_NAMES]
    summary_csv = os.path.join(ROOT_DIR, 'fp_counts_summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(summary_headers)
        writer.writerows(summary_data)
    print(f"Generated Summary CSV: {summary_csv}")
else:
    print("No valid data found across all folders, skipping summary CSV creation.")
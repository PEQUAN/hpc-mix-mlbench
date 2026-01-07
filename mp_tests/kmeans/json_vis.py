import json
import csv

# Define the category display names
CATEGORY_DISPLAY_NAMES = {
    'double': 'FP64',
    'float': 'FP32',
    'half_float::half': 'FP16',
    'flx::floatx<8, 7>': 'BF16',
    'flx::floatx<4, 3>': 'E4M3',
    'flx::floatx<5, 2>': 'E5M2'
}

# List of JSON files to process
json_files = [
    'precision_settings_1.json',
    'precision_settings_2.json',
    'precision_settings_3.json',
    'precision_settings_4.json'
]

# Prepare table data
table_data = []
for file_idx, json_file in enumerate(json_files, 1):
    # Load JSON data from file
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: File {json_file} not found, skipping.")
        continue

    # Process each dictionary in the file
    for dict_idx, entry in enumerate(data, 1):
        row = [file_idx, dict_idx]  # Columns for Precision Setting and Significant Digits
        for key in CATEGORY_DISPLAY_NAMES:
            count = len(entry.get(key, []))  # Get length of list for key, default to 0 if missing
            row.append(count)
        table_data.append(row)

# Prepare headers for CSV
headers = ['Precision Setting', 'Significant Digits'] + [CATEGORY_DISPLAY_NAMES[key] for key in CATEGORY_DISPLAY_NAMES]

# Save to CSV file
with open('fp_counts_all.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(table_data)
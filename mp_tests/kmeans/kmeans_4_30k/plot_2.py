import matplotlib.pyplot as plt
import numpy as np
import json
import os
import time
import csv
from cadnaPromise.run import runPromise

CATEGORY_DISPLAY_NAMES = {
    'double': 'FP64',
    'float': 'FP32',
    'half_float::half': 'FP16',
    'flx::floatx<8, 7>': 'BF16',
    'flx::floatx<4, 3>': 'E4M3',
    'flx::floatx<5, 2>': 'E5M2'
}


CATEGORY_COLORS = {
    'double': '#81D4FAB3',         # Sky Pop Blue
    'float': '#FFAB91B3',          # Candy Coral
    'half_float::half': '#BA68C8B3', # Bubblegum Purple
    'flx::floatx<8, 7>': '#F06292B3', # Strawberry Pink
    'flx::floatx<4, 3>': '#AED581B3', # Apple Green
    'flx::floatx<5, 2>': '#FFF176B3', # Pineapple Yellow
}

def run_experiments(method, digits):
    """Run experiments, collect precision settings, and measure runtime."""
    precision_settings = []
    runtimes = []
    for digit in digits:
        testargs = [
            f'--precs={method}',
            f'--nbDigits={digit}',
            f'--conf=promise.yml',
            '--noParsing',
            '--fp=fp.json'
        ]
        start_time = time.time()
        try:
            result = runPromise(testargs)
            elapsed_time = time.time() - start_time
            if result and isinstance(result, dict):
                cleaned_result = {key: list(value) if isinstance(value, set) else value 
                                for key, value in result.items()}
                precision_settings.append(cleaned_result)
                runtimes.append(elapsed_time)
                print(f"Results for {digit} digits: {cleaned_result}, Runtime: {elapsed_time:.4f} seconds")
            else:
                print(f"Warning: No valid result for {digit} digits")
                precision_settings.append({})
                runtimes.append(0)
        except Exception as e:
            print(f"Error running experiment for {digit} digits: {e}, Runtime: {elapsed_time:.4f} seconds")
            precision_settings.append({})
            runtimes.append(0.0)
    return precision_settings, runtimes

def save_precision_settings(precision_settings, filename='precision_settings_2.json'):
    """Save precision settings to a JSON file."""
    try:
        for setting in precision_settings:
            if not isinstance(setting, dict):
                raise ValueError(f"Invalid data: Expected dict, got {type(setting)}")
            for key, value in setting.items():
                if not isinstance(value, list):
                    raise ValueError(f"Invalid data for {key}: Expected list, got {type(value)}")
        with open(filename, 'w') as f:
            json.dump(precision_settings, f, indent=4)
        print(f"Precision settings saved to {filename}")
    except Exception as e:
        print(f"Error saving precision settings: {e}")
        with open(filename, 'w') as f:
            json.dump([], f)

def save_runtimes_to_csv(digits, runtimes, filename='runtimes2.csv'):
    """Save runtimes and their average to a CSV file."""
    try:
        average_runtime = sum(runtimes) / len(runtimes) if runtimes else 0
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Digit', 'Runtime (seconds)'])
            for digit, runtime in zip(digits, runtimes):
                writer.writerow([digit, f"{runtime:.4f}"])
            writer.writerow(['Average', f"{average_runtime:.4f}"])
        print(f"Runtimes saved to {filename}")
    except Exception as e:
        print(f"Error saving runtimes to CSV: {e}")

def load_precision_settings(filename='precision_settings_2.json'):
    """Load precision settings from a JSON file."""
    if not os.path.exists(filename):
        print(f"Error: {filename} does not exist, regenerating data...")
        precision_settings, _ = run_experiments('whsd', [2, 3, 4, 5])
        save_precision_settings(precision_settings, filename)
        return precision_settings
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Invalid JSON data: Expected list, got {type(data)}")
        for setting in data:
            if not isinstance(setting, dict):
                raise ValueError(f"Invalid JSON data: Expected dict, got {type(setting)}")
            for key, value in setting.items():
                if not isinstance(value, list):
                    raise ValueError(f"Invalid JSON data for {key}: Expected list, got {type(value)}")
        return data
    except Exception as e:
        print(f"Error loading precision settings: {e}")
        print("Regenerating data due to loading error...")
        precision_settings, _ = run_experiments('whsd', [2, 3, 4, 5])
        save_precision_settings(precision_settings, filename)
        return precision_settings

def load_runtimes(filename='runtimes2.csv'):
    """Load runtimes from a CSV file."""
    if not os.path.exists(filename):
        print(f"Error: {filename} does not exist, regenerating data...")
        digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        precision_settings, runtimes = run_experiments('whsd', digits)
        save_runtimes_to_csv(digits, runtimes, filename)
        return runtimes
    try:
        runtimes = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  
            if header != ['Digit', 'Runtime (seconds)']:
                raise ValueError("Invalid CSV header")
            for row in reader:
                if row[0] == 'Average':
                    continue  
                if len(row) != 2:
                    raise ValueError(f"Invalid row format: {row}")
                runtimes.append(float(row[1]))
        return runtimes
    
    except Exception as e:
        print(f"Error loading runtimes: {e}")
        print("Regenerating data due to loading error...")
        digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        precision_settings, runtimes = run_experiments('whsd', digits)
        save_runtimes_to_csv(digits, runtimes, filename)
        return runtimes

def get_categories(precision_settings):
    """Extract unique categories from precision settings, with fallback."""
    categories = set()
    for setting in precision_settings:
        if isinstance(setting, dict):
            categories.update(setting.keys())
    return list(categories) if categories else list(CATEGORY_DISPLAY_NAMES.keys())

def plot_precision_settings(precision_settings, digits, runtimes):
    """Visualize precision settings as a stacked bar chart with observation counts and runtime as a line plot."""
    if not precision_settings:
        print("Error: No precision settings to plot")
        return
    if len(runtimes) != len(digits):
        print(f"Error: Runtime length ({len(runtimes)}) does not match digits length ({len(digits)})")
        return
    if len(precision_settings) != len(digits):
        print(f"Error: Precision settings length ({len(precision_settings)}) does not match digits length ({len(digits)})")
        return
    
    categories = get_categories(precision_settings)
    
    desired_order = [
        'flx::floatx<4, 3>',
        'flx::floatx<5, 2>',
        'flx::floatx<8, 7>',
        'half_float::half',
        'float',
        'double'
    ]
    
    heights = {cat: [] for cat in categories}
    for setting in precision_settings:
        for cat in categories:
            count = len(setting[cat]) if isinstance(setting, dict) and cat in setting else 0
            heights[cat].append(count)
    
    active_categories = [cat for cat in categories if any(heights[cat])]
    if not active_categories:
        print("Error: No non-zero data to plot")
        return
    
    active_categories = sorted(active_categories, key=lambda x: desired_order.index(x) if x in desired_order else len(desired_order))
    
    print("Digits:", digits)
    print("Runtimes:", runtimes)
    print("Precision settings heights:", {cat: heights[cat] for cat in active_categories})
    
    available_styles = plt.style.available
    preferred_style = 'seaborn' if 'seaborn' in available_styles else 'seaborn-v0_8' if 'seaborn-v0_8' in available_styles else 'ggplot'
    try:
        plt.style.use(preferred_style)
        print(f"Using Matplotlib style: {preferred_style}")
    except OSError as e:
        print(f"Warning: Could not use style '{preferred_style}', falling back to 'default'. Error: {e}")
        plt.style.use('default')

    fig, ax = plt.subplots(figsize=(11, 8))
    
    ax2 = ax.twinx()
    
    x_indices = np.arange(len(digits))

    bottom = np.zeros(len(digits))
    # Store bar handles for legend
    bar_handles = []
    bar_labels = []
    for category in active_categories:
        display_name = CATEGORY_DISPLAY_NAMES.get(category, category)
        # Use fixed color from CATEGORY_COLORS, fallback to gray if category not in mapping
        color = CATEGORY_COLORS.get(category, '#808080')
        bars = ax.bar(x_indices, heights[category], bottom=bottom, label=display_name,
                      color=color, width=0.6, edgecolor='white')
        bar_handles.append(bars)
        bar_labels.append(display_name)
        
        for j, (bar_height, bottom_height) in enumerate(zip(heights[category], bottom)):
            if bar_height > 0:
                ax.text(
                    x_indices[j],
                    bottom_height + bar_height / 2,
                    f'{int(bar_height)}',
                    ha='center',
                    va='center',
                    fontsize=15,
                    weight='bold',
                    color='black'
                )
        bottom += np.array(heights[category])

    try:
        runtime_line, = ax2.plot(x_indices, runtimes, color='red', marker='o', linestyle='-', linewidth=2, markersize=8, label='Runtime', zorder=10)
    except Exception as e:
        print(f"Error plotting runtime line: {e}")
        return

    ax.set_ylim(0, max(np.sum([heights[cat] for cat in active_categories], axis=0)) * 1)
    ax2.set_ylim(0, max(runtimes) * 1.5 if runtimes else 1.0)  # Adjust for visibility
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels(digits)

    ax.set_xlim(-0.5, len(digits) - 0.5)

    ax.set_xlabel('Number of required digits', fontsize=16, weight='bold')
    ax.set_ylabel('Number of variables of each type', fontsize=16, weight='bold')
    ax2.set_ylabel('Runtime (seconds)', fontsize=16, weight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.set_title('Precision Settings Distribution with Runtime', fontsize=16, weight='bold', pad=20)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Create legend with explicit order: bars in active_categories order, then runtime
    legend_handles = bar_handles + [runtime_line]
    legend_labels = bar_labels + ['Runtime']
    ax.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=min(len(active_categories) + 1, 6), fontsize=15, frameon=True, edgecolor='black')

    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig('precision2_with_runtime.png', bbox_inches='tight', dpi=300, transparent=False)
    print("Plot saved as precision2_with_runtime.png")
    plt.show()

if __name__ == "__main__":
    method = 'whsd'
    digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


    precision_settings, runtimes = run_experiments(method, digits)
    save_precision_settings(precision_settings)
    save_runtimes_to_csv(digits, runtimes)
    loaded_settings = load_precision_settings()
    loaded_runtimes = load_runtimes()

    if len(loaded_settings) != len(digits):
        print(f"Error: Loaded precision settings length ({len(loaded_settings)}) does not match digits ({len(digits)})")
    elif len(loaded_runtimes) != len(digits):
        print(f"Error: Loaded runtimes length ({len(loaded_runtimes)}) does not match digits ({len(digits)})")
    else:
        plot_precision_settings(loaded_settings, digits, loaded_runtimes)
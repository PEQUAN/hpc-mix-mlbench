import matplotlib.pyplot as plt
import numpy as np
import json
import os
from cadnaPromise.run import runPromise

CATEGORY_DISPLAY_NAMES = {
    'double': 'Double Precision',
    'float': 'Single Precision',
    'half_float::half': 'fp16',
    'flx::floatx<8, 7>': 'bf16',
    'flx::floatx<5, 2>': 'q52',
    'flx::floatx<4, 3>': 'q43'
}

def run_experiments(method, digits):
    """Run experiments and collect precision settings."""
    precision_settings = []
    for digit in digits:
        testargs = [
            f'--precs={method}',
            f'--nbDigits={digit}',
            f'--conf=promise.yml',
            '--noParsing',
            '--fp=fp.json'
        ]
        try:
            result = runPromise(testargs)
            if result and isinstance(result, dict):
                cleaned_result = {key: list(value) if isinstance(value, set) else value 
                                for key, value in result.items()}
                precision_settings.append(cleaned_result)
                print(f"Results for {digit} digits: {cleaned_result}")
            else:
                print(f"Warning: No valid result for {digit} digits")
                precision_settings.append({})
        except Exception as e:
            print(f"Error running experiment for {digit} digits: {e}")
            precision_settings.append({})
    return precision_settings

def save_precision_settings(precision_settings, filename='precision_settings_b.json'):
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
        # Create an empty file to prevent loading corrupted data
        with open(filename, 'w') as f:
            json.dump([], f)

def load_precision_settings(filename='precision_settings_b.json'):
    """Load precision settings from a JSON file."""
    if not os.path.exists(filename):
        print(f"Error: {filename} does not exist, regenerating data...")
        precision_settings = run_experiments('bhsd', [2, 3, 4, 5])
        save_precision_settings(precision_settings, filename)
        return precision_settings
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        # Validate loaded data
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
        precision_settings = run_experiments('bhsd', [2, 3, 4, 5])
        save_precision_settings(precision_settings, filename)
        return precision_settings

def get_categories(precision_settings):
    """Extract unique categories from precision settings, with fallback."""
    categories = set()
    for setting in precision_settings:
        if isinstance(setting, dict):
            categories.update(setting.keys())
    return list(categories) if categories else list(CATEGORY_DISPLAY_NAMES.keys())

def plot_precision_settings(precision_settings, digits):
    """Visualize precision settings as a stacked bar chart."""
    if not precision_settings:
        print("Error: No precision settings to plot")
        return
    
    # Get categories dynamically
    categories = get_categories(precision_settings)
    
    # Define desired legend order (right to left: q43, q52, bf16, fp16, single, double)
    desired_order = [
        'flx::floatx<4, 3>',  # q43
        'flx::floatx<5, 2>',  # q52
        'flx::floatx<8, 7>',  # bf16
        'half_float::half',   # fp16
        'float',              # single
        'double'              # double
    ]
    
    
    heights = {cat: [] for cat in categories}
    for setting in precision_settings:
        for cat in categories:# Filter active categories (non-zero counts) and sort by desired order
            count = len(setting[cat]) if isinstance(setting, dict) and cat in setting else 0
            heights[cat].append(count)
    
    active_categories = [cat for cat in categories if any(heights[cat])]
    if not active_categories:
        print("Error: No non-zero data to plot")
        return
    
    active_categories = sorted(active_categories, key=lambda x: desired_order.index(x) if x in desired_order else len(desired_order))
    
   
    available_styles = plt.style.available # Set up plot style with fallback
    preferred_style = 'seaborn' if 'seaborn' in available_styles else 'seaborn-v0_8' if 'seaborn-v0_8' in available_styles else 'ggplot'
    try:
        plt.style.use(preferred_style)
        print(f"Using Matplotlib style: {preferred_style}")
    except OSError as e:
        print(f"Warning: Could not use style '{preferred_style}', falling back to 'default'. Error: {e}")
        plt.style.use('default')

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))

    x_indices = np.arange(len(digits)) 

    bottom = np.zeros(len(digits))
    for i, category in enumerate(active_categories):
        display_name = CATEGORY_DISPLAY_NAMES.get(category, category)
        ax.bar(x_indices, heights[category], bottom=bottom, label=display_name,
               color=colors[i], width=0.8/len(active_categories), edgecolor='white')
        bottom += np.array(heights[category])

    ax.set_xticks(x_indices)
    ax.set_xticklabels(digits)

    ax.set_xlim(-0.5, len(digits) - 0.5)

    ax.set_xlabel('Number of Digits', fontsize=16, weight='bold')
    ax.set_ylabel('Count of Values', fontsize=16, weight='bold')
    ax.set_title('Precision Settings Distribution', fontsize=16, weight='bold', pad=20)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=min(len(active_categories), 6),
              fontsize=16, frameon=True, edgecolor='black')

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig('precision_kmeans_b.png', bbox_inches='tight', dpi=300, transparent=False)
    print("Plot saved as precision_kmeans_b.png")
    plt.show()

if __name__ == "__main__":
    method = 'cwbsd'
    digits = [2, 4, 6, 8, 10]

    precision_settings = run_experiments(method, digits)
    save_precision_settings(precision_settings)

    loaded_settings = load_precision_settings()
    plot_precision_settings(loaded_settings, digits)
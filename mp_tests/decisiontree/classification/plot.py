import matplotlib.pyplot as plt
import numpy as np
from cadnaPromise.run import runPromise

method = 'bhsd'
digits = [2, 6, 10, 14]
precision_settings = []

if __name__ == "__main__":
    # Run experiments for each digit setting
    for digit in digits:
        testargs = [
            f'--precs={method}',
            f'--nbDigits={digit}',
            '--conf=promise.yml',
            '--noParsing',
            '--fp=fp.json'
        ]
        result = runPromise(testargs)
        print(result)
        precision_settings.append(result)

    # Define all possible categories
    categories = ['double', 'float', 'half_float::half', 'flx::floatx<8, 7>']
    x = digits

    # Initialize heights dictionary for all categories
    heights = {cat: [] for cat in categories}

    # Populate heights, handling missing categories
    for setting in precision_settings:
        for cat in categories:
            # Append the count if category exists in setting, else append 0
            count = len(setting[cat]) if cat in setting else 0
            heights[cat].append(count)

    # Filter out categories with all zero counts (optional, keeps all categories)
    active_categories = [cat for cat in categories if any(heights[cat])]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(precision_settings))

    # Use a color cycle to handle variable number of categories
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    
    for i, category in enumerate(active_categories):
        ax.bar(x, heights[category], bottom=bottom, label=category,
               color=colors[i], width=0.4)
        bottom += np.array(heights[category])

    ax.set_xlabel('Number of Digits')
    ax.set_ylabel('Count of Values')
    ax.set_title('Precision Settings Distribution')
    ax.set_xticks(x)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    plt.tight_layout()
    plt.savefig('precision_dct.png', bbox_inches='tight', dpi=300)
    plt.show()
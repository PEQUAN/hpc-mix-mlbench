import matplotlib.pyplot as plt
import numpy as np
from cadnaPromise.run import runPromise

method = 'bhsd'

digits = [2, 4, 6, 8, 10, 12]
precision_settings = list()


if __name__ == "__main__":
    for digit in digits:
        testargs = ['--precs='+method, '--nbDigits=' + str(digit),
                    '--conf=promise.yml', '--noParsing', '--fp=fp.json']
        
        t = runPromise(testargs)
        print(t)
        precision_settings.append(t)

    categories = ['double', 'float', 'half_float::half', 'flx::floatx<8, 7>']
    x = digits
    
    heights = {cat: [] for cat in categories}
    categories_new = []
    for setting in precision_settings:
        for cat in categories:
            if cat in heights and cat in setting:
                heights[cat].append(len(setting[cat]))
                categories_new.append(cat)
                        
    categories  = set(categories_new)
    
    heights = {height: heights[height] for height in heights}
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bottom = np.zeros(len(precision_settings))
    
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    for i, category in enumerate(categories):
        ax.bar(x, heights[category], bottom=bottom, label=category, 
            color=colors[i], width=0.4)
        
        bottom += heights[category]
    
    ax.set_xlabel('Number of digits')
    ax.set_ylabel('Count of Values')
    ax.set_title('Precision Settings Distribution')
    ax.set_xticks(x)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    
    for i in range(len(precision_settings)):
        total_height = sum(heights[cat][i] for cat in categories if cat in heights)
    
    plt.tight_layout()
    plt.savefig('precision_mlp_reg.png', bbox_inches='tight', dpi=300)
    plt.show()
# coding=utf-8

# This file is part of PROMISE.
#
# 	PROMISE is free software: you can redistribute it and/or modify it
# 	under the terms of the GNU Lesser General Public License as
# 	published by the Free Software Foundation, either version 3 of the
# 	License, or (at your option) any later version.
#
# 	PROMISE is distributed in the hope that it will be useful, but WITHOUT
# 	ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# 	or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General
# 	Public License for more details.
#
# 	You should have received a copy of the GNU Lesser General Public
# 	License along with PROMISE. If not, see
# 	<http://www.gnu.org/licenses/>.
#
# Promise v1 was written by Romain Picot
# Promise v2 has been written from v1 by Thibault Hilaire and Sara Hoseininasab
# Promise v3 has been written from v2 by Thibault Hilaire, Fabienne JÉZÉQUEL and Xinye Chen
#   Promise v3 enables version check, pip installing, arbitrary precision, etc., features.
# 	  Sorbonne Universitéx, LIP6 (Computing Science Laboratory), Paris, France. 
#     Contact: thibault.hilaire@lip6.fr, fabienne.jezequel@lip6.fr, xinyechenai@gmail.com
#
# 	contain the entry function, called to run Promise
#
# 	© Thibault Hilaire and Fabienne JÉZÉQUEL, April 2024



"""
\U0001f918 cadnaPromise \U0001f918

Usage:
    promise (-h | --help)
    promise (-v | --version)
    promise --precs=<strs> [options]

Options:
    -h --help                     Show this screen.
    --version                     Show version.
    --precs=<strs>                Set the precision following the built-in or cutomized precision letters [default: sd]
    --conf CONF_FILE              Get the configuration file [default: promise.yml]
    --fp FPT_FILE                 Get the file for floating point number format [default: fp.json]
    --output OUTPUT               Set the path of the output (where the result files are put)
    --verbosity VERBOSITY         Set the verbosity (betwen 0  and 4 for very low level debug) [default: 1]
    --log LOGFILE                 Set the log file (no log file if this is not defined)
    --verbosityLog VERBOSITY      Set the verbosity of the log file
    --debug                       Put intermediate files into `debug/` (and `compileErrors/` for compilation errrors) and display the execution trace when an error comes
    --run RUN                     File to be run
    --compile COMMAND             Command to compile the code
    --files FILES                 List of files to be examined by Promise (by default, all the .cc files)
    --nbDigits DIGITS             General required number of digits
    --path PATH                   Set the path of the project (by default, the current path)
    --pause                       Do pause between steps
    --parsing                     Parse the C file (without this, __PROMISE__ are replaced and that's all)
    --auto                        enable auto-instrumentation of source code
    --relError THRES              use criteria of precision relative error less than THRES instead of number of digits
    --noCadna                     will not use cadna, reference result computed in (non-stochastic) double precision
    --alias ALIAS                 Allow aliases (examples "g++=g++-14") [default:""]
    --CC        				  Set compiler for C program [default: g++]
    --CXX                         Set compiler for C++ program [default: g++]
    --plot                        Enable plotting of results [default: 1]
"""


import os
import json
from os.path import join
import sys

from docopt import docopt
from .utils import parseOptions, PromiseError, getYMLOptions, Timing, pause, commaAnd
from .utils import getFPM, sort_precs, update_types, get_version

from .logger import PrLogger


# types handle by cadnaPromise
_typeNames = {'b':'bfloat16', 
              'h': 'Half', 
              's': 'Single', 
              'd': 'Double', 
              'q': 'Quad', 
              'o': 'Octuple'
              }

_types = {'b': 'flx::floatx<8, 7>', 
          'h': 'half_float::half', 
          's': 'float', 
          'd': 'double', 
          'q': 'float128',
          'o': 'flx::floatx<19, 236>',
         }



def loadCADNA():
    if 'CADNA_PATH' in os.environ:
        print(os.environ['CADNA_PATH'])
    
    else:
        curr_loc = os.path.dirname(os.path.realpath(__file__))
        print(curr_loc+'/cadna')


def runPromise(argv=None):
    
    """This function is registered (in setup.py) as an entry_point
    argv is used for the unit tests"""
    from .prfile import PrFile
    from .promise import Promise
    # reset the logger and get a new instance
    logger = PrLogger()
    
    # reset the handlers, in case of running runPromise several times (otherwise, the log files are still open)
    
    displayTrace = False
    EARLY_SROP = False
        
    try:
        if argv is None:
            if '--help' in sys.argv[1:] or len(sys.argv[1:]) == 0:
                print(__doc__)
                return
        else:
            if '--help' in argv or len(argv) == 0:
                print(__doc__)
                return

        if argv is not None:
            if '--version' in argv or '--v' in argv:
                EARLY_SROP = True

        else:
            if '--version' in sys.argv[1:] or '--v' in sys.argv[1:]:
                EARLY_SROP = True

        logger.reset()
        args = docopt(__doc__, argv=sys.argv[1:] if argv is None else argv)
        displayTrace = args['--debug']

        logger.configureLogger(args)   # configure the logger

        if EARLY_SROP:
            logger.message("cadnaPromise version " + get_version(
                os.path.dirname(os.path.realpath(__file__))+'/__init__.py') + ' (cadna version 3.1.12)')
     
            logger.message("Copyright (c) 2025, GNU General Public License v3.0")
            logger.message(
                f"This work was supported by the France 2030 NumPEx Exa-MA (ANR-22-EXNU-0002) project managed by the French National Research Agency (ANR)."
                )
            
            return 
        

        options = getYMLOptions(args)                                           # get the options from the yml file
        method, path, files, run, nbDigits, _, compileLines, outputPath, typeCustom, alias = parseOptions(options)    # parse the options

        fpfmt = getFPM(args)
        method = sort_precs(method, fpfmt)

        types, typeNames = update_types(_types, _typeNames, fpfmt)
        compiler = 'g++'

        if isinstance(alias, dict):
            if alias == {}:
                curr_loc = os.path.dirname(os.path.realpath(__file__))

                cachePath = "/cache"
                if os.path.exists(curr_loc + cachePath):
                    if os.path.isfile(curr_loc + cachePath + '/CXX.txt'):
                        with open(curr_loc+cachePath+"/CXX.txt", "r") as file:
                            compiler = file.read().replace('\n', '')
                            print('check compilers:', compiler)
            
            elif alias.get('g++', False):
                compiler = alias['g++']

        else:
            compiler = alias
            alias = {}
            
        if compiler != 'g++' and compiler is not None:
            alias['g++'] = compiler
        

        logger.message("\U0001f918 cadnaPromise \U0001f918")
        logger.message("Using the compiler: {}".format(compiler))

        PrFile.setCustom(typeCustom)
        compileErrorPath = join(path, 'compileErrors') if args['--debug'] else None
        tempPath = join(path, 'debug') if args['--debug'] else None

        # run with timing
        with Timing() as timing:
            # create Promise object
            pr = Promise(path, files, run, nbDigits, compileLines, parsing=args['--parsing'], alias=alias)

            # display general infos
            logger.info("We are working with %d file%s and %d different types" %
                        (pr.nbFiles, ('' if pr.nbFiles < 2 else 's'), pr.nbVariables))
            logger.info(pr.expectation())

            # debug the files
            if args['--debug']:
                pr.exportParsedFiles(tempPath)

            # get the cadna reference
            highest = types['q'] if 'q' in method else types['d']
            # print("highest:", highest)
            pr.changeSameType(highest)
            logger.step("Get a reference result with cadna (%s)" % highest)
            pr.compileAndRun(tempPath, cadna=True)
            if args['--pause']:
                pause()

            # try with the highest precision
            logger.step("Check with highest format (%s)" % typeNames[method[-1]])
            pr.changeSameType(types[method[-1]])
            
            if not pr.compileAndRun(tempPath):
                pr.changeSameType(highest)
                raise PromiseError("You should lower your expectation, it doesn't work with " + typeNames[method[-1]])
            
            if args['--pause']:
                pause()
            

            # do the Delta-Debug passes ('s','d' and then 'h','s' when method is 'hsd' for example)
            for lo, hi in reversed(list(zip(method, method[1:]))):
                logger.step("Delta-Debug %s/%s" % (typeNames[lo], typeNames[hi]))
                res = pr.runDeltaDebug(types[lo], types[hi], tempPath, args['--pause'], compileErrorPath)
                # stop if the DeltaDebug is not successful
                if not res:
                    break

        # export the output
        if argv is None:
            pr.exportFinalResult(outputPath)


    except PromiseError as e:
        logger.error(e, exc_info=displayTrace)

    else:
        if timing:
            # from collections import Counter
            # display the number of each type
            # count = Counter(pr.typesDict.values())  # count the nb of each type (result is a dictionary type:nb)
            # li = ["%dx %s" % (v, k) for k, v in count.items()]
            _output =  pr.setPerType()
            output = {types[i]:_output[types[i]] for i in fpfmt if types[i] in _output}
            output = dict(reversed(list(output.items())))
        
            li = ["%dx %s" % (len(output[i]), i) for i in output]

            logger.message("The final result contains %s.", commaAnd(li))
            logger.debug("Final types:\n" + pr.strResult())

            # display the stats
            logger.message("It tooks %.2fs", timing.timing)
            logger.message("\U0001F449 %d compilations (%d failed) for %.2fs", *pr.compilations)
            logger.message("\U0001F449 %d executions   (%d failed) for %.2fs", *pr.executions)

            logger.reset()

            return output
        

    logger.reset()








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
    import time
    precision_settings = []
    runtimes = []
    for digit in digits:
        testargs = [
            f'--precs={method}',
            f'--nbDigits={digit}',
            f'--conf=promise.yml',
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

def save_precision_settings(precision_settings, filename='precision_settings.json'):
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

def save_runtimes_to_csv(digits, runtimes, filename='runtimes.csv'):
    """Save runtimes and their average to a CSV file."""
    import csv
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

def load_precision_settings(filename='precision_settings.json'):
    """Load precision settings from a JSON file."""
    
    if not os.path.exists(filename):
        print(f"Error: {filename} does not exist, regenerating data...")
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

def load_runtimes(filename='runtimes.csv'):
    """Load runtimes from a CSV file."""
    import csv
    if not os.path.exists(filename):
        print(f"Error: {filename} does not exist, regenerating data...")
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


def get_categories(precision_settings):
    """Extract unique categories from precision settings, with fallback."""
    categories = set()
    for setting in precision_settings:
        if isinstance(setting, dict):
            categories.update(setting.keys())
    return list(categories) if categories else list(CATEGORY_DISPLAY_NAMES.keys())



def plot_precision_settings(precision_settings, digits, runtimes):
    import numpy as np
    import matplotlib.pyplot as plt
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
    plt.savefig('precision_with_runtime.png', bbox_inches='tight', dpi=300, transparent=False)
    print("Plot saved as precision_with_runtime.png")
    plt.show()



def run_experiment_and_plot(argv=None):
    args = docopt(__doc__, argv=sys.argv[1:] if argv is {} else argv,  version=get_version(os.path.dirname(os.path.realpath(__file__))+'/__init__.py'))
    method = args['--precs']

    if ',' in args['--nbDigits']:
        parts = args['--nbDigits'].split(',')
        digits = [int(d) for d in parts] if args['--nbDigits'] else None

    elif ':' in args['--nbDigits']:
        parts = args['--nbDigits'].split(':')
        digits = list(range(int(parts[0]), int(parts[1]) + 1)) if args['--nbDigits'] else None

    else:
        digits = [args['--nbDigits']]

    print(f"Digits to test: {digits}")
    plot = args['--plot']

    if method is None:
        method = 'cbsd'
            
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
        if plot == True:
            plot_precision_settings(loaded_settings, digits, loaded_runtimes)


if __name__ == "__main__":
    runPromise()




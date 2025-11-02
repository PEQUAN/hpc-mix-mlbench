# MIX-HPC-ML benchmarks


## Benchmark Run

To run the mixed-precision benchmarks by PROMISE, one need to go to the folder ``mp_tests``, and use the command:

```bash
cd mp_tests
```

A **Bash automation script** to run `plot_1.py` to `plot_4.py` across multiple experiment folders, with support for:

- Running **experiments** (`precision_settings_1.json`, `promise.yml`)
- **Plotting** results
- Selective execution of **specific folders**

---

The folder structure:


```text
project/
├── exp_set1/
│   ├── plot_1.py
│   ├── plot_2.py
│   ├── plot_3.py
│   ├── plot_4.py
│   ├── precision_settings_1.json   ← required (generated or pre-existing)
│   └── promise.yml                 ← required (generated or pre-existing)
├── exp_set2/
│   └── (same files as above)
├── incomplete_set/                 ← will be skipped (missing files)
├── run_benchmarks.sh                    ← this script
└── README.md                       ← this file


> **Each valid folder must contain**:
> - `plot_1.py`, `plot_2.py`, `plot_3.py`, `plot_4.py`
> - `precision_settings_1.json`
> - `promise.yml`

---


1. Save the script as `run_benchmarks.sh` in your project root.
2. Make it executable:

```bash
chmod +x run_benchmarks.sh
```


```bash
./run_benchmarks.sh <run_experiments> <run_plotting> [folder1 folder2 ...]
```

### ⚙️ Customization & Advanced Features

| 🧩 **Feature** | ✏️ **How to Modify** |
|:----------------|:--------------------|
| **Search Depth** | 🔍 Change `find . -maxdepth 2` → `-maxdepth 3` for deeper search, or remove `-maxdepth` for unlimited depth. |
| **Python Path** | 🐍 Replace `python3` with `python` or a specific interpreter path, e.g. `/path/to/venv/bin/python`. |
| **Add Logging** | 🧾 Redirect all output: `./run_plots.sh ... &> log.txt` (saves stdout and stderr). |
| **Parallel Runs** | ⚡ Install GNU Parallel: `sudo apt install parallel`. Then replace the loop with:  <br>```bash<br>export RUN_EXPERIMENTS RUN_PLOTTING<br>parallel -j4 run_folder ::: "${TARGET_FOLDERS[@]}"<br>``` |
| **More Files** | ➕ Add entries to the `missing=()` loop, e.g. `other_file.txt`. |

> 💡 **Tip:** Combine multiple tweaks for more flexible automation (e.g., deeper search + parallel execution).



### 🧭 Usage Guide

| 🖥️ **Command** | 📘 **Description** |
|:----------------|:------------------|
| `./run_benchmarks.sh` | 🧪 Run **experiments + plots** in all valid folders *(default)* |
| `./run_benchmarks.sh 1 0` | ⚙️ Run **only experiments** (skip plots) in all folders |
| `./run_benchmarks.sh 0 1` | 📊 Run **only plots** (uses saved data) in all folders |
| `./run_benchmarks.sh 1 1 setA setB` | 🎯 Run **both** in only `setA` and `setB` |
| `./run_benchmarks.sh n y exp_set1` | 🧩 Skip experiments, **plot only** in `exp_set1` *(short form)* |
| `./run_benchmarks.sh false true results/v1 results/v3` | 🎨 Plot only in **two specific folders** (`results/v1`, `results/v3`) |

> 💡 **Tip:**  
> - Arguments follow the pattern:  
>   `./run_benchmarks.sh [run_experiments] [run_plots] [optional_folder_names...]`  
> - Accepted values:  
>   `1 / true / y` = yes | `0 / false / n` = no



### Generate Summary

After running all experiments, one can enenerate the number of floating point types for each precision settings:

```bash
python json_counts_sum.py

```


### 🧩 Common Issues & Solutions

| ⚠️ **Issue** | 🛠️ **Solution** |
|:--------------|:----------------|
| `precision_settings_1.json` or `promise.yml` not found | 📁 Ensure both files exist in the same folder as `plot_*.py`. The script checks automatically, but regenerate them if missing (set `run_experiments=1`). |
| `[Errno 2] No such file or directory` | 📂 Likely a path issue — run `cd` into the correct directory. If it persists, add `SCRIPT_DIR = Path(__file__).parent` in your Python script to use absolute paths. |
| **No folders found** | 🔍 Verify folder names and spelling. You can also increase `-maxdepth` in your search command for deeper directory scanning. |
| **Permission denied** | 🔑 Run `chmod +x run_benchmarks.sh` to make the script executable. |
| **Python errors** | 🧠 Check the script output. Make sure required libraries like `matplotlib` and `yaml` are installed in your environment. |

> 💡 **Tip:** Run `bash -x run_benchmarks.sh` for verbose debugging if you need to trace what the script is doing.


## License


This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
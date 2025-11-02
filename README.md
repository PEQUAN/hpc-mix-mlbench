# MIX-HPC-ML benchmarks


# Benchmark RUn

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


1. Save the script as `run_plots.sh` in your project root.
2. Make it executable:

```bash
chmod +x run_benchmarks.sh
```


```bash
./run_plots.sh <run_experiments> <run_plotting> [folder1 folder2 ...]
```


### 🧭 Usage Guide

| 🖥️ **Command** | 📘 **Description** |
|:----------------|:------------------|
| `./run_plots.sh` | 🧪 Run **experiments + plots** in all valid folders *(default)* |
| `./run_plots.sh 1 0` | ⚙️ Run **only experiments** (skip plots) in all folders |
| `./run_plots.sh 0 1` | 📊 Run **only plots** (uses saved data) in all folders |
| `./run_plots.sh 1 1 setA setB` | 🎯 Run **both** in only `setA` and `setB` |
| `./run_plots.sh n y exp_set1` | 🧩 Skip experiments, **plot only** in `exp_set1` *(short form)* |
| `./run_plots.sh false true results/v1 results/v3` | 🎨 Plot only in **two specific folders** (`results/v1`, `results/v3`) |

> 💡 **Tip:**  
> - Arguments follow the pattern:  
>   `./run_plots.sh [run_experiments] [run_plots] [optional_folder_names...]`  
> - Accepted values:  
>   `1 / true / y` = yes | `0 / false / n` = no
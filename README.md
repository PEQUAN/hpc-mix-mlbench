# HPC-MIX benchmarks

## Setup

To properly use the benchmark tool, one would need to install ``cadnaPromise`` in advance. To install, use:
```bash
pip install cadnaPromise
```
see [cadnaRromise](cadnaPromise/) in detail. 

To add the code for benchmarking, add a project folder inside directory ``mp_tests`` including data aa well as executable code, and properly configured file ``promise.yml`` as well as floating point type format ``fp.json``. 



To set up the results and plots, one can go the the folder ``run_settings``, and cutomized the plots and settings for the files  ``run_settings_{k}_py`` (you can copy multiple files there and will all be run). Once completed, go the folder  ``mp_tests``, and run the ``sync_run_settings.sh`` in ``mp_tests`` directory, then the ``run_settings_{k}_py`` in ``run_settings`` will be broadcasted to each subfolder. 
The script ``sync_run_settings.sh`` is useful for automating the synchronization of experiment or run settings files across multiple folders for benchmarking, the usage is below:

```bash
bash sync_run_settings.sh [options]
```

Options:
* --delete or -d: Execute Step 1 (delete files) for clearning. Default: enabled if no options.
* --copy or -c: Execute Step 2 (copy files). Default: enabled if no options.



## Benchmark Run

To run the mixed-precision benchmarks by PROMISE, one need to go to the folder ``mp_tests``, and use the command:

```bash
cd mp_tests
```


A **Bash automation script** to run `run_setting_1.py` to `run_setting_4.py` exisit across multiple experiment folders, with support for:

- Running **experiments** (marked executable code, `promise.yml`, `fp.json`)
- **Plotting** results
- Selective execution of **specific folders**

---

We stored the numerical algorithm to be benchmarked in setA, setB, ...., and more. The folder structure: 


```text
projects/
├── setA/                     <- Each set contains a test for a single benchmark
│   ├── run_setting_1.py
│   ├── ......
│   ├── run_setting_k.py
│   ├── C/C++ code to be examined
│   ├── fp.json               <- floating point format defined by users or left as default
│   ├── prec_setting_1.json   <- results by PROMISE
│   ├── ......
│   ├── prec_setting_k.json   <- results by PROMISE
│   └── promise.yml                 <- required (generated or pre-existing)
├── setB/
│   └── (same files as above)
├── incomplete_set/                 <- will be skipped (missing files)
├── run_benchmarks.sh               <- main benchmark script


> **Each valid folder must contain**:
> - `run_setting_1.py`, `run_setting_2.py`, `run_setting_3.py`, `run_setting_4.py`
> - `prec_setting_1.json`, `prec_setting_2.json`, `prec_setting_3.json`, `prec_setting_4.json`
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
| `./run_benchmarks.sh` | 🧪 Run **experiments + plots** **sequential** in all valid folders *(default)* |
| `./run_benchmarks.sh 1 0` | ⚙️ Run **only experiments** (skip plots) in all folders |
| `./run_benchmarks.sh 0 1` | 📊 Run **only plots** (uses saved data) in all folders |
| `./run_benchmarks.sh 1 1 setA setB` | 🎯 **Sequential:** run **both** in only `setA` and `setB` |
| `./run_benchmarks.sh n y setA` | 🧩 Skip experiments, **plot only** in `setA` *(short form)* |
| `./run_benchmarks.sh 0 1 --parallel` | ⚡ **Parallel:** runs **plots only** in all folders. |
| `./run_benchmarks.sh 1 0 --parallel` | ⚡ **Parallel:** runs **experiments only** in all folders. |
| `./run_benchmarks.sh 1 1 --parallel` | ⚡ **Parallel:** runs **experiments and plots** in all folders. |
| `./run_benchmarks.sh 1 1 setA setB --parallel` | 🔀 **Parallel:** runs **experiments + plots** only in `setA` and `setB`. |




> 💡 **Tip:**  
> - Arguments follow the pattern:  
>   `./run_benchmarks.sh [run_experiments] [run_plots] [optional_folder_names...]`  
> - Accepted values:  
>   `1 / true / y` = yes | `0 / false / n` = no


In the end, one can quickly check their visualization in the plots folder via:

```bash
cd mp_tests
bash organize_plots.sh
```

### Generate Summary

After running all experiments, one can enenerate the number of floating point types for each precision settings:

```bash
python json_counts_sum.py

```


## License


This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

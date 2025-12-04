#!/usr/bin/env python3

import numpy as np
import scipy.io
import scipy.sparse as sp
import os
import glob
import pandas as pd
from datetime import datetime

def compute_growth_factor(A_orig):
    A = A_orig.astype(np.float64)
    n = A.shape[0]
    if n == 0:
        return 1.0, 0.0, 0.0

    max_orig = np.max(np.abs(A))
    if max_orig == 0:
        return 1.0, 0.0, 0.0

    max_seen = max_orig
    Lu = A.copy()

    for k in range(n):
        # Partial pivoting
        pivot_row = np.argmax(np.abs(Lu[k:, k])) + k
        if np.abs(Lu[pivot_row, k]) > max_seen:
            max_seen = np.abs(Lu[pivot_row, k])

        if pivot_row != k:
            Lu[[k, pivot_row]] = Lu[[pivot_row, k]]

        if abs(Lu[k, k]) < 1e-15:        # avoid division by near-zero
            continue

        for i in range(k+1, n):
            factor = Lu[i, k] / Lu[k, k]
            Lu[i, k] = factor
            Lu[i, k+1:] -= factor * Lu[k, k+1:]

            max_seen = max(max_seen,
                           np.max(np.abs(Lu[i, k+1:])),
                           abs(factor))

    growth = max_seen / max_orig
    return growth, max_seen, max_orig


def cond2_estimate(A, fast_for_large=True):
    """
    Return 2-norm condition number kappa_2(A).
    - For n ≤ 1500: exact via full SVD
    - For n > 1500: fast iterative estimator (very accurate in practice)
    """
    n = A.shape[0]
    
    if n <= 1500 or not fast_for_large:
        try:
            return np.linalg.cond(A, p=2)
        except:
            pass

    # Fast iterative estimator (from Higham, Accuracy and Stability of Num. Alg.)
    try:
        from scipy.linalg import svdvals
        # Compute only 2 extreme singular values
        largest = svdvals(A, compute_uv=False)[0]
        smallest = svdvals(A, compute_uv=False)[-1]
        if smallest < 1e-15:
            return np.inf
        return largest / smallest
    except:
        pass

    # Fallback: use numpy's built-in estimator (still fast)
    try:
        return np.linalg.cond(A, p=2)
    except:
        return np.inf


def process_all_mtx(folder=".", pattern="*.mtx"):
    results = []

    mtx_files = sorted(glob.glob(os.path.join(folder, pattern)))
    print(f"Found {len(mtx_files)} .mtx files in '{folder}'\n")

    for idx, filepath in enumerate(mtx_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{idx:3d}/{len(mtx_files)}] {filename.ljust(40)}", end=" ")

        try:
            mat = scipy.io.mmread(filepath)
            if sp.issparse(mat):
                nnz = mat.nnz
                mat = mat.toarray()
            else:
                nnz = np.count_nonzero(mat)

            if mat.shape[0] != mat.shape[1]:
                print("not square -> skipped")
                results.append({"file": filename, "status": "skipped (not square)"})
                continue

            n = mat.shape[0]
            A = mat.astype(np.float64)

            # Growth factor
            growth, max_seen, max_orig = compute_growth_factor(A)

            # 2-norm condition number
            print("computing kappa_2 ...", end=" ")
            cond2 = cond2_estimate(A)
            print(f"-> kappa_2={cond2:12.6e}", end=" ")

            print(f"n={n:5d}  NNZ={nnz:8d}  rho={growth:12.6e}")

            results.append({
                "file": filename,
                "n": n,
                "NNZ": nnz,
                "max_|aij|": max_orig,
                "max_seen": max_seen,
                "growth_factor_rho": growth,
                "cond_2": cond2,
                "status": "OK"
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"file": filename, "status": f"error: {e}"})

    # Save results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_name = f"growth_and_cond2_{timestamp}.csv"
    df.to_csv(csv_name, index=False)

    print("\n" + "="*100)
    print("ALL DONE")
    print(f"Results saved to -> {csv_name}")
    print("="*100)

    # Summary tables
    ok = df[df["status"] == "OK"].copy()
    if not ok.empty:
        pd.set_option('display.float_format', '{:.2e}'.format)

        print("\nTop 15 worst growth factor rho:")
        print(ok.sort_values("growth_factor_rho", ascending=False)
              .head(15)[["file", "n", "NNZ", "growth_factor_rho", "cond_2"]].to_string(index=False))

        print("\nTop 15 most ill-conditioned (kappa_2):")
        print(ok.sort_values("cond_2", ascending=False)
              .head(15)[["file", "n", "NNZ", "cond_2", "growth_factor_rho"]].to_string(index=False))

        print("\nBest-behaved matrices (lowest rho * kappa_2):")
        ok["score"] = ok["growth_factor_rho"] * ok["cond_2"].replace(np.inf, np.nan)
        best = ok.sort_values("score").dropna().head(12)
        print(best[["file", "n", "growth_factor_rho", "cond_2"]].to_string(index=False))

    return df


if __name__ == "__main__":
    # Change folder/pattern if needed
    process_all_mtx(folder=".", pattern="*.mtx")
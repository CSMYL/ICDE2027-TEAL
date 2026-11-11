#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = HERE

CONT_COLS = ["carat", "depth", "table", "x", "y", "z", "price"]
CAT_COLS = ["cut", "color", "clarity"]


def standardize_by_train(
    df_train: pd.DataFrame, df_test: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Standardize continuous features using train statistics (mean/std) and apply to train/test.
    Returns standardized DataFrames and (col -> (mean, std)) records.
    """
    stats = {}
    train = df_train.copy()
    test = df_test.copy()

    for col in cols:
        mu = float(train[col].mean())
        sigma = float(train[col].std(ddof=0))
        if sigma == 0.0:
            sigma = 1.0
        stats[col] = (mu, sigma)
        for df_ in (train, test):
            df_[col] = (df_[col] - mu) / sigma

    return train, test, stats


def main() -> None:
    summary_rows = []

    for exp_id in (1, 2, 3):
        train_path = os.path.join(EXP_DIR, f"train_exp{exp_id}.csv")
        test_path = os.path.join(EXP_DIR, f"test_exp{exp_id}.csv")
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            raise FileNotFoundError(f"Missing train/test files for exp{exp_id}: {train_path}, {test_path}")

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        n_train, n_test = len(df_train), len(df_test)
        n_total = n_train + n_test
        train_pct = 100.0 * n_train / n_total
        test_pct = 100.0 * n_test / n_total

        std_train, std_test, _ = standardize_by_train(df_train, df_test, CONT_COLS)

        df_shift = pd.concat([std_train, std_test], axis=0, ignore_index=True)

        out_name = f"shift{exp_id}.csv"
        out_path = os.path.join(EXP_DIR, out_name)
        df_shift.to_csv(out_path, index=False)

        summary_rows.append(
            {
                "Shift": f"shift{exp_id}",
                "Exp": f"Exp {exp_id}",
                "Train Count": n_train,
                "Test Count": n_test,
                "Train %": f"{train_pct:.2f}%",
                "Test %": f"{test_pct:.2f}%",
            }
        )

        print(
            f"shift{exp_id}: train={n_train}, test={n_test}, "
            f"train%={train_pct:.2f}%, test%={test_pct:.2f}% -> wrote {out_name}"
        )

    df_summary = pd.DataFrame(summary_rows)
    headers = list(df_summary.columns)
    rows = df_summary.astype(str).values.tolist()

    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(vals: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    header_line = fmt_row(headers)
    align_line = "| " + " | ".join((":---").ljust(widths[i]) for i in range(len(headers))) + " |"
    body_lines = [fmt_row(r) for r in rows]
    md_table = "\n".join([header_line, align_line] + body_lines)

    md_path = os.path.join(EXP_DIR, "shifts_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Diamonds OOD Shifts Train/Test Ratio Summary\n\n")
        f.write(md_table + "\n")

    print(f"\nGenerated shift1/2/3 CSVs and Markdown summary: {md_path}")


if __name__ == "__main__":
    main()

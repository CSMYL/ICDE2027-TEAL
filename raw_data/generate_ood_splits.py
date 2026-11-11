from __future__ import annotations

import os
import numpy as np
import pandas as pd


INPUT_FILE = "diamonds.csv"
# Note: diamonds.csv is headerless; column names are assigned below:
COLUMN_NAMES = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "price"]
OUTPUT_DIR = "diamonds_ood_experiments"
RANDOM_SEED = 42


def _pearson_corr(x: pd.Series, y: pd.Series) -> float:
    return float(x.corr(y, method="pearson"))

def _to_markdown_table(df: pd.DataFrame) -> str:
    """Minimal markdown table renderer (avoid pandas.to_markdown dependency on tabulate)."""
    headers = list(df.columns)
    rows = df.astype(str).values.tolist()

    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(vals: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    header_line = fmt_row(headers)
    align_line = "| " + " | ".join((":---").ljust(widths[i]) for i in range(len(headers))) + " |"
    body_lines = [fmt_row(r) for r in rows]
    return "\n".join([header_line, align_line] + body_lines)


def make_ood_split(
    df_all: pd.DataFrame,
    group_a_mask: pd.Series,
    *,
    seed: int,
    experiment_name: str,
    score_mean_spec: tuple[str, str] | None,
    corr_spec: tuple[str, str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, int]:
    """
    Use N=min(|A|,|B|) to create mutually exclusive Train 90/10 and Test 10/90 splits:
      - Train: A_train=round(0.9N), B_train=N-A_train
      - Test : A_test=N-A_train,   B_test=A_train
    """
    rng = np.random.default_rng(seed)

    df_a = df_all.loc[group_a_mask].copy()
    df_b = df_all.loc[~group_a_mask].copy()
    n_a, n_b = len(df_a), len(df_b)
    if n_a == 0 or n_b == 0:
        raise ValueError(f"{experiment_name}: one group is empty (A={n_a}, B={n_b})")

    N = min(n_a, n_b)
    n_a_train = int(round(0.9 * N))
    n_a_test = N - n_a_train
    n_b_train = n_a_test
    n_b_test = n_a_train

    a_ids = rng.choice(df_a["row_id"].to_numpy(), size=N, replace=False)
    b_ids = rng.choice(df_b["row_id"].to_numpy(), size=N, replace=False)

    rng.shuffle(a_ids)
    rng.shuffle(b_ids)

    a_train_ids = set(a_ids[:n_a_train])
    a_test_ids = set(a_ids[n_a_train:])
    b_train_ids = set(b_ids[:n_b_train])
    b_test_ids = set(b_ids[n_b_train:])

    train_ids = a_train_ids | b_train_ids
    test_ids = a_test_ids | b_test_ids
    assert train_ids.isdisjoint(test_ids), f"{experiment_name}: train/test overlap detected"

    train_df = df_all[df_all["row_id"].isin(train_ids)].copy()
    test_df = df_all[df_all["row_id"].isin(test_ids)].copy()

    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    def _summary(split_df: pd.DataFrame) -> dict:
        split_a_pct = 100.0 * float(split_df["row_id"].isin(a_ids).mean())
        split_b_pct = 100.0 - split_a_pct

        row = {
            "Experiment": experiment_name,
            "Split": "Train" if split_df is train_df else "Test",
            "Sample Count": len(split_df),
            "Group A %": f"{split_a_pct:.1f}%",
            "Group B %": f"{split_b_pct:.1f}%",
        }

        if score_mean_spec is not None:
            col, label = score_mean_spec
            row[label] = f"{float(split_df[col].mean()):.3f}"
        else:
            row["Target Mean"] = "-"

        if corr_spec is not None:
            xcol, ycol = corr_spec
            corr = _pearson_corr(split_df[xcol], split_df[ycol])
            row["Correlation (Clarity vs Price)"] = f"{corr:+.3f}"
        else:
            row["Correlation (Clarity vs Price)"] = "-"

        return row

    train_row = _summary(train_df)
    test_row = _summary(test_df)
    return train_df, test_df, train_row, test_row, N


def main() -> None:
    print("=" * 80)
    print("Diamonds OOD/Causal Learning Experiment Data Generation")
    print("=" * 80)

    print("\n[1] Loading data...")
    df_raw = pd.read_csv(INPUT_FILE, header=None, names=COLUMN_NAMES)
    print(f"Raw data: {len(df_raw)} rows, {df_raw.shape[1]} cols")
    print(f"Columns: {list(df_raw.columns)}")

    required_cols = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "price"]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns in {INPUT_FILE}: {missing}")

    print("\n[2] Deduplication (must be done before splitting)...")
    before = len(df_raw)
    df = df_raw.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"Before: {before} | After: {len(df)} | Removed duplicates: {removed}")

    df["row_id"] = np.arange(len(df), dtype=np.int64)

    print("\n[3] Encoding check (ordinal encoding convention from diamonds_mapping.csv)...")
    print(f"cut unique: {sorted(df['cut'].unique().tolist())}")
    print(f"color unique: {sorted(df['color'].unique().tolist())}")
    print(f"clarity unique: {sorted(df['clarity'].unique().tolist())}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []

    # Exp 1: Cut Shift
    # A: cut in {Premium(3), Ideal(4)}  <=> cut >= 3
    print("\n" + "=" * 80)
    print("Exp 1: Cut Shift")
    print("=" * 80)
    exp1_a = df["cut"] >= 3
    train1, test1, r1_train, r1_test, N1 = make_ood_split(
        df,
        exp1_a,
        seed=RANDOM_SEED + 1,
        experiment_name="Exp 1 (Cut Shift)",
        score_mean_spec=("cut", "Cut Mean"),
        corr_spec=None,
    )
    train1.drop(columns=["row_id"]).to_csv(os.path.join(OUTPUT_DIR, "train_exp1.csv"), index=False)
    test1.drop(columns=["row_id"]).to_csv(os.path.join(OUTPUT_DIR, "test_exp1.csv"), index=False)
    rows += [r1_train, r1_test]
    print(f"Sample size N=min(|A|,|B|)={N1} -> train/test each {N1} rows")

    # Exp 2: Color Shift
    # A: color in {D(6),E(5),F(4)} <=> color >= 4
    print("\n" + "=" * 80)
    print("Exp 2: Color Shift")
    print("=" * 80)
    exp2_a = df["color"] >= 4
    train2, test2, r2_train, r2_test, N2 = make_ood_split(
        df,
        exp2_a,
        seed=RANDOM_SEED + 2,
        experiment_name="Exp 2 (Color Shift)",
        score_mean_spec=("color", "Color Mean"),
        corr_spec=None,
    )
    train2.drop(columns=["row_id"]).to_csv(os.path.join(OUTPUT_DIR, "train_exp2.csv"), index=False)
    test2.drop(columns=["row_id"]).to_csv(os.path.join(OUTPUT_DIR, "test_exp2.csv"), index=False)
    rows += [r2_train, r2_test]
    print(f"Sample size N=min(|A|,|B|)={N2} -> train/test each {N2} rows")

    # Exp 3: Simpson's Paradox
    # median split: carat (size) and clarity (clarity_score)
    # A (spurious/negative): (small & high) OR (big & low)
    # B (causal/positive):  (small & low)  OR (big & high)
    print("\n" + "=" * 80)
    print("Exp 3: Simpson's Paradox")
    print("=" * 80)
    median_carat = float(df["carat"].median())
    median_clarity = float(df["clarity"].median())
    print(f"median(carat)={median_carat:.4f} | median(clarity_score)={median_clarity:.4f}")

    small = df["carat"] < median_carat
    high = df["clarity"] >= median_clarity
    exp3_a = (small & high) | ((~small) & (~high))

    train3, test3, r3_train, r3_test, N3 = make_ood_split(
        df,
        exp3_a,
        seed=RANDOM_SEED + 3,
        experiment_name="Exp 3 (Simpson)",
        score_mean_spec=None,
        corr_spec=("clarity", "price"),
    )
    train3.drop(columns=["row_id"]).to_csv(os.path.join(OUTPUT_DIR, "train_exp3.csv"), index=False)
    test3.drop(columns=["row_id"]).to_csv(os.path.join(OUTPUT_DIR, "test_exp3.csv"), index=False)
    rows += [r3_train, r3_test]
    print(f"Sample size N=min(|A|,|B|)={N3} -> train/test each {N3} rows")

    # Verification table
    print("\n" + "=" * 80)
    print("Verification Table (Markdown)")
    print("=" * 80)

    df_summary = pd.DataFrame(rows)
    col_order = [
        "Experiment",
        "Split",
        "Sample Count",
        "Group A %",
        "Group B %",
        "Cut Mean",
        "Color Mean",
        "Correlation (Clarity vs Price)",
    ]
    for c in col_order:
        if c not in df_summary.columns:
            df_summary[c] = "-"
    df_summary = df_summary[col_order]

    md_table = _to_markdown_table(df_summary)
    print("\n" + md_table)

    df_summary.to_csv(os.path.join(OUTPUT_DIR, "verification_table.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "verification_table.md"), "w", encoding="utf-8") as f:
        f.write(md_table + "\n")

    design_md = f"""# Diamonds OOD / Causal Learning (Soft Shifts / Selection Bias) Experiment Design

## Data Source & Encoding

- Input: `{INPUT_FILE}` (ordinal encoded per `diamonds_mapping.csv`)
- cut: Fair(0) < Good(1) < Very Good(2) < Premium(3) < Ideal(4)
- color: J(0) < I(1) < H(2) < G(3) < F(4) < E(5) < D(6)
- clarity: I1(0) < SI2(1) < SI1(2) < VS2(3) < VS1(4) < VVS2(5) < VVS1(6) < IF(7)

## Deduplication (prevents data leakage)

Deduplicated before splitting: {before} rows -> {len(df)} rows ({removed} duplicate rows removed).

## Global Sampling Strategy (9:1 <-> 1:9)

For each experiment, define A/B groups and use **N = min(|A|, |B|)** for mutually exclusive splits:

- Train: 90% A + 10% B
- Test : 10% A + 90% B
- Strictly mutually exclusive: verified via `row_id` that train/test have no overlap.

Both train and test have N samples each, with A/B ratios strongly reversed by design.

## Exp 1: Cut Shift (single-feature shift)

- A (high cut): cut ∈ {{Premium(3), Ideal(4)}} (i.e., cut ≥ 3)
- B (low cut): cut ∈ {{Fair, Good, Very Good}} (i.e., cut ≤ 2)

## Exp 2: Color Shift (single-feature shift)

- A (high color): color ∈ {{D(6), E(5), F(4)}} (i.e., color ≥ 4)
- B (low color): color ∈ {{G, H, I, J}} (i.e., color ≤ 3)

## Exp 3: Simpson's Paradox (spurious correlation reversal)

Median split:

- median(carat) = {median_carat:.4f}
- median(clarity_score) = {median_clarity:.4f}

Groups:

- A (spurious/simple pattern): (small carat & high clarity) OR (big carat & low clarity)
- B (causal/hard pattern): (small carat & low clarity) OR (big carat & high clarity)

Validation metric: Pearson corr(clarity_score, price) should show significant Train vs Test difference (ideally a negative->positive flip).

## Output Files

Output directory: `{OUTPUT_DIR}/`

- `train_exp1.csv`, `test_exp1.csv`
- `train_exp2.csv`, `test_exp2.csv`
- `train_exp3.csv`, `test_exp3.csv`
- `verification_table.csv`, `verification_table.md`

Verification table:

{md_table}
"""

    with open(os.path.join(OUTPUT_DIR, "EXPERIMENT_DESIGN.md"), "w", encoding="utf-8") as f:
        f.write(design_md)

    print(f"\nDone. Output: {OUTPUT_DIR}/ (6 CSVs + verification_table + EXPERIMENT_DESIGN.md)")


if __name__ == "__main__":
    main()

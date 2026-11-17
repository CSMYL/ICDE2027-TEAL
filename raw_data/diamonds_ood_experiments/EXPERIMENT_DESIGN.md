# Diamonds OOD / Causal Learning (Soft Shifts / Selection Bias) Experiment Design

## Data Source & Encoding

- Input: `diamonds.csv` (headerless; column names assigned by `generate_ood_splits.py`)
- cut: Fair(0) < Good(1) < Very Good(2) < Premium(3) < Ideal(4)
- color: J(0) < I(1) < H(2) < G(3) < F(4) < E(5) < D(6)
- clarity: I1(0) < SI2(1) < SI1(2) < VS2(3) < VS1(4) < VVS2(5) < VVS1(6) < IF(7)

## Deduplication (prevents data leakage)

Deduplicated before splitting: 53940 rows -> 53794 rows (146 duplicates removed).

## Global Sampling Strategy (9:1 <-> 1:9)

For each experiment, define A/B groups and use **N = min(|A|, |B|)** for mutually exclusive splits:

- Train: 90% A + 10% B
- Test : 10% A + 90% B
- Strictly mutually exclusive: verified via `row_id` that train/test have no overlap.

Both train and test have N samples each, with A/B ratios strongly reversed by design.

## Exp 1: Cut Shift (single-feature shift)

- A (high cut): cut ∈ {Premium(3), Ideal(4)} (i.e., cut ≥ 3)
- B (low cut): cut ∈ {Fair, Good, Very Good} (i.e., cut ≤ 2)

## Exp 2: Color Shift (single-feature shift)

- A (high color): color ∈ {D(6), E(5), F(4)} (i.e., color ≥ 4)
- B (low color): color ∈ {G, H, I, J} (i.e., color ≤ 3)

## Exp 3: Simpson's Paradox (spurious correlation reversal)

Median split:

- median(carat) = 0.7000
- median(clarity_score) = 3.0000

Groups:

- A (spurious/simple pattern): (small carat & high clarity) OR (big carat & low clarity)
- B (causal/hard pattern): (small carat & low clarity) OR (big carat & high clarity)

Validation metric: Pearson corr(clarity_score, price) should show significant Train vs Test difference (ideally a negative->positive flip).

## Output Files

Output directory: `diamonds_ood_experiments/`

- `train_exp1.csv`, `test_exp1.csv`
- `train_exp2.csv`, `test_exp2.csv`
- `train_exp3.csv`, `test_exp3.csv`
- `verification_table.csv`, `verification_table.md`

Verification table:

| Experiment          | Split | Sample Count | Group A % | Group B % | Cut Mean | Color Mean | Correlation (Clarity vs Price) |
| :---                | :---  | :---         | :---      | :---      | :---     | :---       | :---                           |
| Exp 1 (Cut Shift)   | Train | 18558        | 90.0%     | 10.0%     | 3.406    | nan        | -                              |
| Exp 1 (Cut Shift)   | Test  | 18558        | 10.0%     | 90.0%     | 1.770    | nan        | -                              |
| Exp 2 (Color Shift) | Train | 26051        | 90.0%     | 10.0%     | nan      | 4.604      | -                              |
| Exp 2 (Color Shift) | Test  | 26051        | 10.0%     | 90.0%     | nan      | 2.299      | -                              |
| Exp 3 (Simpson)     | Train | 19355        | 90.0%     | 10.0%     | nan      | nan        | -0.416                         |
| Exp 3 (Simpson)     | Test  | 19355        | 10.0%     | 90.0%     | nan      | nan        | +0.443                         |

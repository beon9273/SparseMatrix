# scripts

## generate.py

Reads a CSV matrix and prints the corresponding `SparseMat<>` declaration.

```bash
python3 generate.py <matrix.csv>
```

Each row in the CSV is a matrix row. Zero values are treated as structural zeros and excluded from the index pack. Non-zero values appear as constructor arguments in row-major order.

**Example** — `sample.csv` (6×8, ~30% density):

```bash
python3 generate.py sample.csv
# SparseMat<double, 6, 8, 1, 2, 5, 6, 8, 11, 13, 16, 18, 32, 33, 34, 35, 38, 42, 43> m(3.2, ...);
```

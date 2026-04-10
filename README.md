# Paper Reproductions

This repository stores focused reproductions of research papers. Each paper lives under
`papers/` with its own code, experiment outputs, and notes.

## Current contents

- `papers/pisa_reproduction`: a stricter reproduction-oriented implementation for
  *Permutation-Invariant Set Autoencoders with Fixed-Size Embeddings for Multi-Agent Learning*
  (PISA, AAMAS 2023)

## Repository conventions

- keep each paper in a dedicated folder under `papers/`
- store runnable code, generated figures, metrics, and paper-specific notes together
- keep legacy reports if they are useful for comparison, but label them clearly

## Environment

Install the dependencies from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

Then follow the paper-specific instructions in
[`papers/pisa_reproduction/README.md`](papers/pisa_reproduction/README.md).

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent-Based Model simulation of "Talent vs Luck" based on Pluchino, Biondo & Rapisarda (2018). N agents with normally-distributed talent interact with lucky/unlucky events in a 2D world over 80 time steps (~40 working years), demonstrating how randomness drives wealth inequality.

## Language and Dependencies

- Python 3.10+ (uses `int | None` union type syntax)
- numpy, matplotlib (Agg backend for headless rendering)
- No package manager config — install dependencies manually or via conda

## How to Run

```bash
cd talent_vs_luck
python talent_vs_luck.py    # Single run, prints stats, saves 2 PNG plots
python run_100.py            # 100-run aggregate statistics, saves 1 PNG plot (several minutes)
```

## Architecture

**`TalentVsLuckSimulation`** class (`talent_vs_luck/talent_vs_luck.py`):
- `__init__` — Creates N agents with random 2D positions, normal-distributed talent (mean=0.6, std=0.1, clipped [0,1]), equal initial capital (10.0). Uses `numpy.random.default_rng(seed)` for reproducibility.
- `_generate_events()` — Places lucky/unlucky event positions randomly each step.
- `_check_interactions()` — Vectorized distance calculation (events vs agents within perception radius). Lucky events: capital doubles with probability = talent. Unlucky events: capital halves unconditionally.
- `run()` — Main simulation loop (default 80 steps), returns capital history array.

`talent_vs_luck/run_100.py` imports `TalentVsLuckSimulation` and runs it 100 times, aggregating wealth distribution statistics and correlation metrics.

Visualization functions `plot_results()` and `plot_capital_evolution()` in `talent_vs_luck.py` produce 2-panel and 4-panel matplotlib figures saved as PNGs.

## Conventions

- All comments, docstrings, and console output are in Chinese; variable/function names are in English.
- All simulation parameters are hardcoded in `if __name__ == "__main__"` blocks.
- Output PNG files are written to the current working directory (ignored by `.gitignore`).
- No test suite, linter, or build system is configured.

## Gotchas

- `run_100.py` uses `sys.path.insert` to import from the same directory — it must be run from `talent_vs_luck/` directly.
- All generated plots (`*.png`) are gitignored; they won't appear in `git status`.

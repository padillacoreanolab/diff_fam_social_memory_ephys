# CLAUDE.md

Ephys analysis for a social-memory project: Trodes + Phy recordings → spike & LFP
analysis, with behavior extracted from BORIS / ECU / SLEAP. Experiment design &
phase context live in Notion ("Social Memory Ephys Pilot 2").

## Map

**Reusable libraries** (stable, tested — import these, don't copy their logic):
- `spike/spike_analysis/` — `SpikeCollection`, `SpikeRecording`, decoders, firing rate, normalization, population & single-cell. See `spike/CLAUDE.md`.
- `lfp/lfp_analysis/` — `LFP_collection`, `LFP_recording`, connectivity, event extraction, preprocessor, plotting. See `lfp/README.md`.
- `behavior/` — BORIS / ECU extraction + epoch tools + SLEAP helpers. See `behavior/CLAUDE.md`.
- `trodes/`, `vid_helper_fxns/` — raw export & video conversion. See `trodes/CLAUDE.md`.

**Analysis** (per-experiment notebooks — the actual science, one folder per Pilot 2 phase):
- `pilot2/habit_dishabit_phase1/`, `cups_phase4/`, `only_subjects/`, `object_control/`, `rehouse/` — each has its own `CLAUDE.md` (what the experiment is + notebook run-order).
- `data_analysis/pilot1/` — older pilot, same patterns.
- `*.Rmd` GLM scripts + `r_stuff/*.rds` — R stats layer.

**Do not touch — reference only, may be outdated:**
- `old_notebooks/`, `other_peoples_sutff/`, and any `*/archived/` dir.
- Never edit these, never cite them as the current pattern. Read only to understand history.

## Non-negotiable gotchas

- **Working dir = repo root.** Imports and (un)pickling are relative to the repo root, NOT the notebook's location. VSCode must have Jupyter Notebook File Root = `${workspaceFolder}`. Breaking this silently breaks pickles. Details: `README.md`.
- **numpy stays 1.x.** `spikeinterface==0.100.6` hard-crashes the kernel on numpy 2.x. Don't bump numpy without bumping spikeinterface. Details: `lfp/README.md`.
- **GPU LFP** needs `SPECTRAL_CONNECTIVITY_ENABLE_GPU=true` (set inline atop LFP notebooks). See `lfp/README.md`.

## Workflow

- Env: `conda env create -f ephys_env.yml && conda activate ephys_env`. LFP extras: `lfp/requirements.txt` (+ `requirements-gpu.txt` for GPU).
- Tests: `pytest spike/tests` and `pytest lfp/tests`. Add tests when changing library code.
- Assumes Trodes/Phy default naming (`*merged.rec/`, `.time/`, `phy/`). See "What do you need" in `README.md`.
- Notebooks import from libraries, e.g. `from spike.spike_analysis.spike_collection import SpikeCollection`.

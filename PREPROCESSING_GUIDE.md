# Preprocessing Guide: Raw Behavior + Spikes → Ephys Object

A step-by-step walkthrough for turning **already-scored BORIS**, **already-tracked SLEAP**,
**already-spike-sorted Phy** data plus your **organized-but-not-yet-JSON timestamps** into a
`SpikeCollection` whose recordings carry behavior `event_dict`s aligned to the neural clock.

This guide is written for the state you are in right now: BORIS + SLEAP exist, spikes are
sorted, and the Trodes timestamps have been pulled into their own folder but have **not** been
folded into a `.json` yet. Step 2 is the piece you still need to run.

> Read this alongside the top-level [CLAUDE.md](CLAUDE.md) and [README.md](README.md). The
> library code cited here lives in [behavior/](behavior/), [spike/spike_analysis/](spike/spike_analysis/),
> and [trodes/](trodes/). Example notebooks live under [pilot2/](pilot2/).

---

## 0. Mental model — why any of this is necessary

There are **two independent clocks** you must reconcile before a spike time and a behavior bout
mean anything together:

| Clock | Unit | Zeroed at | Where it comes from |
|---|---|---|---|
| **Ephys clock** | 20 kHz samples | "play" (recording onset) | Phy `spike_times.npy`; Trodes `first_timestamp` |
| **Video/behavior clock** | video frames | "stream" (camera stream onset) | `.videoTimeStamps`; BORIS frames; SLEAP frames |

The camera starts counting at **stream** onset, but ephys data is zeroed at **play** onset.
Everything you extract from behavior (BORIS bouts, SLEAP poses, mouse-in/out markers) is on the
video clock and must be shifted onto the ephys clock. The entire alignment reduces to one line:

```
play_indexed_seconds = stream_indexed_seconds − (first_timestamp / 20000)
# equivalently, in ms:  ... − (first_timestamp / 20)
```

where `first_timestamp` is the ephys sample index at which recording began (read from the Trodes
`*.timestamps.dat` header) and `20000` is the headstage sampling rate. **This is what the
timestamps `.json` in Step 2 exists to precompute.** Everything downstream is bookkeeping around
this subtraction.

The final product — a behavior `event_dict` — is a dict like
`{"novel": np.array([[start_ms, stop_ms], ...]), "familiar": ...}`, one `(n_bouts, 2)` array of
**play-indexed milliseconds** per behavior, attached to each `SpikeRecording`.

---

## 1. Prerequisites & directory assumptions

**Environment** (see [README.md](README.md)):
```bash
conda env create -f ephys_env.yml
conda activate ephys_env
```

**⚠️ Working directory = repo root, always.** Set VSCode **Jupyter: Notebook File Root** to
`${workspaceFolder}`. Imports (`from spike.spike_analysis...`) and *especially* (un)pickling are
relative to the repo root, not the notebook's folder. Break this and pickles load into the wrong
namespace and silently corrupt. Do **not** bump numpy past 1.x (`spikeinterface==0.100.6` crashes
the kernel on numpy 2.x).

**What you must already have on disk** (naming conventions are load-bearing — the loaders glob on
these exact suffixes):

- **Phy / spikes:** one folder per recording ending in `merged.rec`, each containing a `phy/`
  subfolder with `cluster_group.tsv`, `spike_times.npy`, `spike_clusters.npy`. Grouped under a
  `sorted/` parent.
- **Trodes timestamps:** the `*.timestamps.dat` files (from a `-time` export) and the
  `*.videoTimeStamps` files — this is the "separate folder" you already organized.
- **BORIS:** aggregated event tables (`.csv` or `.tsv`), one per recording/subject.
- **SLEAP:** `.h5` prediction files (ideally identity-corrected).

---

## 2. Build the timestamps `.json` ← **the step you still need to run**

You have `.timestamps.dat` and `.videoTimeStamps` organized in a folder. This step reads them,
computes the play-indexed arrays, and writes a JSON that every behavior-alignment step consumes.

### 2a. (If not already done) export timestamps from raw `.rec`

If you only have raw `*merged.rec` folders and no `.dat`/`.videoTimeStamps` yet, run a Trodes
export first. See [pilot2/rehouse/trodes_export.ipynb](pilot2/rehouse/trodes_export.ipynb):

```python
import trodes.read_exported as trodes
trodes.trodes_extract(
    trodes_directory=r"...\Trodes_2-5-2_Windows64",  # your Trodes install
    data=r"...\<phase>\data",                        # folder of *merged.rec
    mode="-raw -time",                               # -time produces *.timestamps.dat
)
```
`-time` yields the `*.timestamps.dat` (containing `first_timestamp`). Add `-dio`/`-analogio` only
if you also recorded operant/ECU events (see Step 6, optional). Requires a local Trodes install.

The two low-level readers you'll rely on live in
[trodes/read_exported.py](trodes/read_exported.py):
- `read_trodes_extracted_data_file(path)` — reads any `.dat`; header exposes `first_timestamp`.
- `readCameraModuleTimeStamps(path)` — reads a `.videoTimeStamps` file → per-frame times in
  **seconds** (parses the `Clock rate:` header, then `np.fromfile(...) / clock_rate`).

### 2b. Build and write the JSON

The canonical, in-repo example is
[trodes/timestamps_json_extraction.ipynb](trodes/timestamps_json_extraction.ipynb). The recipe:

1. **Collect `first_timestamp` per recording** by walking the `.dat` files:
   ```python
   first_timestamp_dict = {}
   for root, dirs, files in os.walk(dat_path):
       for file in files:
           if file.endswith(".dat"):
               ts = tr.read_trodes_extracted_data_file(os.path.join(root, file))
               key = "_".join(file.split("_")[:2])      # e.g. "11_cups"
               first_timestamp_dict[key] = int(ts["first_timestamp"])
   ```

2. **For each `.videoTimeStamps`, compute the play-indexed array** and pack a per-recording dict
   with three keys — `stream_indexed_array`, `play_indexed_array`, `first_timestamp`:
   ```python
   videotsarray = tr.readCameraModuleTimeStamps(videotsfile)          # seconds, stream-indexed
   first_ts = first_timestamp_dict[key]
   play_indexed = videotsarray - first_ts / 20000                     # ← the core subtraction
   play_indexed_dict[rec] = {
       "stream_indexed_array": videotsarray,
       "play_indexed_array":   play_indexed,
       "first_timestamp":      first_ts,
   }
   ```

3. **Convert numpy → lists and dump** (JSON can't hold ndarrays):
   ```python
   def convert_numpy_to_list(obj):
       if isinstance(obj, np.ndarray): return obj.tolist()
       if isinstance(obj, dict):  return {k: convert_numpy_to_list(v) for k, v in obj.items()}
       if isinstance(obj, list):  return [convert_numpy_to_list(x) for x in obj]
       return obj
   with open("pilot2/<phase>/timestamps_dict_<phase>.json", "w") as f:
       json.dump(convert_numpy_to_list(play_indexed_dict), f, indent=2)
   ```

A working reference output already exists in-repo:
[pilot2/object_control/timestamps_dict_object_control.json](pilot2/object_control/) (plus its
lightweight pickle `first_timestamps_object_control`).

> **Two parallel conventions exist in this codebase — pick one and be consistent:**
> - **Full JSON** (`timestamps_dict_*.json`): arrays + `first_timestamp` per recording. Consumed
>   by `behavior_dict_creation.ipynb` / `object_behavior.ipynb`. **Prefer this.**
> - **Lightweight pickle** (`first_timestamps_*`): just `{key: first_timestamp}`. Consumed by
>   `only_subjects/spike_collection_sleap_boris_dict_creation.ipynb`, which re-reads the raw
>   `.videoTimeStamps` separately. Older style.
>
> A second variant, [vid_helper_fxns/timestamp_extraction.ipynb](vid_helper_fxns/), maps videos
> to recordings via an Excel sheet (`habit_dishabit_vid_map.xlsx`) instead of by filename key —
> see Step 7. ⚠️ Known bug in that notebook: it *writes* the JSON to a `homecage_social_ephys_vids`
> path but the consumer *reads* from a `cum_socialmem_ephys` path — reconcile the paths if you use it.

---

## 3. Extract BORIS bouts — **Mac vs Windows is the key fork**

All BORIS logic lives in [behavior/boris_extraction.py](behavior/boris_extraction.py). There is
**no `if platform == "mac"` anywhere.** The OS difference is entirely about **which columns BORIS
writes**, and you pick the matching function. Auto-detect at runtime — never hard-code the OS.

### 3a. Read the file (extension picks the delimiter)

```python
if file_path.endswith(".csv"):
    df = pd.read_csv(file_path)
elif file_path.endswith(".tsv"):
    df = pd.read_csv(file_path, sep="\t")
```

### 3b. Choose the extractor by inspecting the columns

| | **Windows / "none-mac" scored** | **Mac scored** |
|---|---|---|
| Detect via | `"Image index start" in df.columns` | else branch (has `FPS` + `Start (s)`) |
| Timing columns | `Image index start` / `Image index stop` (frame numbers) | `Start (s)` / `Stop (s)` (seconds) + an `FPS` column |
| Function | `get_behavior_bouts_frame(...)` | `get_behavior_bouts_fps(...)` |
| Frame conversion | uses frame index directly → `cameratimestamps[idx]` | `round(Start(s)) * FPS` → frame → `cameratimestamps[idx]` |

The canonical branch, from
[pilot2/habit_dishabit_phase1/behavior_dict_creation.ipynb](pilot2/habit_dishabit_phase1/behavior_dict_creation.ipynb):

```python
first_timestamp  = data_dict[recording_name]["first_timestamp"]
cameratimestamps = np.array(data_dict[recording_name]["camera timestamps"])  # stream-indexed sec

if "Image index start" in df.columns:                 # ── Windows / none-mac
    sniffs = boris.get_behavior_bouts_frame(df, cameratimestamps, first_timestamp,
                                            subject, behavior, min_iti, min_bout)
    ins_frames  = np.array(df[df["Behavior"] == "mice back in"]["Image index start"])
    outs_frames = np.array(df[df["Behavior"] == "mice taken out"]["Image index start"])
else:                                                  # ── Mac
    sniffs = boris.get_behavior_bouts_fps(df, cameratimestamps, first_timestamp,
                                          subject, behavior, min_iti, min_bout)
    # in/out epoch markers derived from Start (s) instead of Image index
```

Both `_frame` and `_fps` internally: map frames → seconds via `cameratimestamps`, subtract
`first_timestamp / 20000`, and multiply by 1000 → they **return `(n_bouts, 2)` arrays already in
play-indexed milliseconds**, exactly the shape `SpikeRecording.event_dict` wants.

- `get_behavior_bouts_frame` also forward-fills a missing `Image index stop` with the next row's
  `Image index start` (`.shift(-1)`).
- `get_behavior_bouts_fps` finds the FPS column by scanning names for the substring `"FPS"`, and
  clamps the final stop frame if it overruns the timestamp array by one.

### 3c. The "plain" extractor (no camera alignment)

When you're slicing by scored seconds and don't need frame→ephys alignment (e.g. the cups
analyses, ethograms), use `get_behavior_bouts(df, subject, behavior, min_iti, min_bout)`. It reads
`Start (s)`/`Stop (s)` directly and returns ms. Example:
[pilot2/cups_phase4/cups2_behavior_analysis.ipynb](pilot2/cups_phase4/) slices `baseline_df` /
`familiarization_df` / `last10min_df` by `Start (s)` then calls `get_behavior_bouts` per epoch.

> `subject` and `behavior` are **lists**. `min_iti` merges bouts closer than N seconds; `min_bout`
> drops bouts shorter than N seconds. Both default to 0. Thresholding is done by
> `behavioral_epoch_tools.threshold_bouts`.

---

## 4. Extract SLEAP tracks (optional, for pose-based epochs)

SLEAP helpers are in [behavior/sleap_helper_fxns.py](behavior/sleap_helper_fxns.py). A
`sleap_vid` wrapper reads the `.h5` (`track_names`, `tracks`, `node_names`, `point_scores`) into
`.locations` / `.track_dict` / `.node_dict`. To put pose data on the ephys ms grid, resample with:

```python
info = sleap.sleap_to_1ms(sleap_array, timestamps_ms, start_ms, stop_ms)
```

where `timestamps_ms` is the **play-indexed** camera timestamps in ms
(`play_indexed_array * 1000`). It forward-fills per-frame poses into 1 ms bins so pose and spikes
share one grid. SLEAP poses are typically carried in a **separate** `sleap_dict`, not inside the
spike `event_dict`.

---

## 5. Assemble the per-recording `behavior_dict` and pickle it

Loop your recordings, build each behavior's `(n,2)` ms array via Step 3, and assemble a dict
keyed by **recording folder name** (the same string `SpikeRecording.name` will have, e.g.
`23_CNF_merged.rec`), then pickle it. This decouples the (slow, fiddly) behavior extraction from
spike-object construction. Fullest worked example:
[pilot2/only_subjects/spike_collection_sleap_boris_dict_creation.ipynb](pilot2/only_subjects/spike_collection_sleap_boris_dict_creation.ipynb).

```python
behavior_dicts = {}   # {rec_name: {"novel": arr_ms, "familiar": arr_ms, "cagemate": arr_ms, ...}}
# ... fill via Step 3 per recording ...
pickle_this(behavior_dicts, "pilot2/<phase>/behavior_dict_by_frames.pkl")
```

**Mouse in/out epochs** (when the experimenter adds/removes a stimulus mouse) are aligned the same
way: `(cameratimestamps[frames] − first_timestamp / 20000) * 1000`. **Baseline epochs** are often
synthesized with `behavioral_epoch_tools.random_event_generator`.

Subject IDs are **parsed from the folder name**, not read from a sheet:
```python
subject = str(int(rec_name.split("_")[0]) / 10)   # "23_CNF_merged.rec" → "2.3"
subject_dict = {rec: str(int(rec.split("_")[0]) / 10) for rec in behavior_dicts}
```

---

## 6. Build the `SpikeCollection` and attach behavior

Core classes: [spike/spike_analysis/spike_collection.py](spike/spike_analysis/spike_collection.py)
and [spike/spike_analysis/spike_recording.py](spike/spike_analysis/spike_recording.py).

**Construction walks a folder for `*merged.rec` dirs**, builds a `SpikeRecording` from each
`phy/` subfolder (reading `cluster_group.tsv` + `spike_times.npy` + `spike_clusters.npy`, dropping
`noise` clusters), and **excludes recordings with no `good` units** (printed to console).

The canonical minimal recipe (see
[pilot2/habit_dishabit_phase1/spike_object_creation.ipynb](pilot2/habit_dishabit_phase1/spike_object_creation.ipynb)):

```python
import spike.spike_analysis.spike_collection as collection

# 1) Create — point at the folder that CONTAINS the *merged.rec folders
col = collection.SpikeCollection(r"...\<phase>\spike_data\sorted")

# 2) Load the behavior dict you pickled in Step 5
behavior_dicts = unpickle_this(r"pilot2/<phase>/behavior_dict_by_frames.pkl")

# 3) Attach subject + event_dict to each recording (key by recording.name)
for recording in col.recordings:
    recording.set_subject(str(int(recording.name.split("_")[0]) / 10))
    recording.set_event_dict(behavior_dicts[recording.name])

# 4) (Optional) compute firing rates / spiketrains
col.analyze(timebin=100, ignore_freq=0.5, smoothing_window=500)   # ms

# 5) Save
col.save_collection(r"pilot2/<phase>/<phase>_collection")
```

You can also pass `subject_dict` / `event_dict` at construction
(`SpikeCollection(path, event_dict=..., subject_dict=...)`) and they'll be attached during
`make_collection()`. `analyze()` then validates via `__all_set__()` that every recording has a
subject + event_dict, that all event_dicts share the same keys, and that every event array is
`(n, 2)` — it prints **"All set to analyze"** when the collection is coherent.

> **Persistence is HDF5 + JSON, not pickle.** `save_collection(path)` writes
> `path/spike_collection.json` (metadata) + `path/recordings/<name>.h5` (+ `.json`) per recording.
> Reload with `SpikeCollection.load_collection(path)`. Analysis products (spiketrains, firing
> rates) are intentionally **not** saved — rerun `analyze()` after loading. `first_timestamp` is
> stored in the H5 metadata only if you set `recording.first_timestamp` yourself (the classes
> never assign it — it comes from your Step 2 data if you want it there).

> **⚠️ API drift across notebooks:** current source uses `col.recordings` (a list). Older
> notebooks like `cups2_spike_collection.ipynb` use `col.collection`. Use `.recordings`.

**Optional — ECU / DIO events** (operant chamber inputs, not video): if you exported `-dio`, build
hardware-event dicts with `behavior/ecu_extraction.py::rec_to_ecu` / `merged_rec_to_box`. These
are zeroed to recording onset with the same `first_timestamp` subtraction. Requires two hand-made
maps (see Step 7).

---

## 7. External files you must supply — and what's missing from the repo

The pipeline reads several files that **BORIS / SLEAP / Phy / Trodes do not produce for you**.
These are hand-made and, in several cases, **not checked into the repo** (they live under personal
`Dropbox`/`Documents` paths hard-coded in the notebooks). You will need to recreate or relocate
them for a new dataset:

| File | Read by | What it must contain | Status |
|---|---|---|---|
| `habit_dishabit_vid_map.xlsx` | [vid_helper_fxns/timestamp_extraction.ipynb](vid_helper_fxns/) | columns `videoTimeStamps` (video-ts filename) + `Merged.rec` (matching recording) | **❌ not in repo** (personal `Documents` path) |
| `habit_dishabit_timestamps.json` | [behavior_dict_creation.ipynb](pilot2/habit_dishabit_phase1/behavior_dict_creation.ipynb) | per-rec `vid_name`, `camera timestamps`, `camera timestamps play indexed`, `first_timestamp` | **❌ not in repo** — regenerate via Step 2 |
| `timestamps_dict_object_control.json` + `first_timestamps_object_control` | object_control notebooks | see Step 2 | ✅ in repo ([pilot2/object_control/](pilot2/object_control/)) — use as a template |
| BORIS aggregated CSV/TSV dir | Step 3 notebooks | columns `Subject`, `Behavior`, `Start (s)`, `Stop (s)`; +`Image index start/stop` (Windows) or `FPS` (Mac) | BORIS output, **dir is a personal path** |
| SLEAP `.h5` predictions dir | Step 4 | HDF5 `track_names`, `tracks`, `node_names`, `point_scores` | SLEAP output, **dir is a personal path** |
| `.videoTimeStamps` / `.timestamps.dat` dir | Step 2 | Trodes camera + ephys timestamps | Trodes output, **dir is a personal path** |

**Mappings that are hard-coded inside notebooks (not files) — you must transcribe these by hand
for a new dataset:**

- **Cup/agent-order + colors:** `cup_dict` (subject → agent-per-cup) and `color_dict` in
  [pilot2/cups_phase4/cups2_behavior_analysis.ipynb](pilot2/cups_phase4/).
- **ECU channel + box maps:** `box_to_ecu_dict` (box → `{dio_ECU_DinX: event}`) and
  `rec_to_box_dict` (`*_merged.rec` → box) in `behavior/ecu_extract.ipynb`, consumed by
  `behavior/ecu_extraction.py::merged_rec_to_box`.
- **One-off timing overrides**, e.g. `times = [605, 1203, 1864]` for a specific cups file.

There is **no** brain-region or channel-map spreadsheet anywhere — region/condition/subject
information is carried entirely by **recording-name parsing** and these inline dicts.

---

## 8. Validation plots — confirm each stage before moving on

Generate these as you go. They are the difference between "the code ran" and "the alignment is
correct."

### After SLEAP load — tracking sanity
- **`plot_tracks(sleap_obj, node="Center")`** (in `behavior_dict_creation.ipynb`): left = X-vs-Y
  arena trajectory per track, right = X-position over frame index.
  **✅ Good:** two spatially separated mouse tracks that stay in arena bounds with no teleport
  jumps. Jumps = SLEAP identity swaps → fix before continuing.
- **"crowded frames only" track plot** (only_subjects notebook): re-plots tracks restricted to
  frames where both mice are detected — an explicit identity check during interaction.

### After BORIS extraction — bouts land in the right epochs
- **`behavioral_epoch_tools.plot_event_bars(eventA, event_dict, eventB, title)`**: horizontal-bar
  timeline; originals on top, derived (overlap/split/combine) rows below.
  **✅ Good:** derived overlap bars sit exactly where A and B intersect.
- **`create_ethogram(...)`** ([pilot2/habit_dishabit_phase1/ethogram.ipynb](pilot2/habit_dishabit_phase1/ethogram.ipynb)):
  rectangle-patch ethogram straight from BORIS `Subject/Behavior/Start (s)/Stop (s)`.
  **✅ Good:** colored blocks only during expected times, nothing past max `Stop (s)`.
- **"Social Interactions Timeline" / "Line Segments"** (`behavior_dict_creation.ipynb`,
  `cups2_behavior_analysis.ipynb`): one row per recording, each bout a colored segment by epoch.
  **✅ Good:** bouts fall inside the correct experiment-epoch windows across the session; bout
  density is plausible per animal.

### After alignment — the critical BORIS↔SLEAP↔ephys check
- **`plot_ethogram()`** (only_subjects): per-subject 1×3 panel (cagemate/novel/familiar) from
  run-length-encoded per-frame labels; prints `fps`, `np.unique(behavior_array)`, nonzero count.
  **✅ Good:** nonzero behavior only inside the agent epochs; sane fps.
- **`plot_pairwise_ethogram()`** (only_subjects) — **the money plot:** two ethogram rows plus a
  third panel overlaying both mice's SLEAP X-position on the same time axis.
  **✅ Good:** sniff/interaction bouts coincide with the two position traces converging. If bouts
  and position don't line up, your `first_timestamp` subtraction or video-map is wrong.
- **1 ms-resample sanity trace** after `sleap_to_1ms()`: `plt.plot(info['locations_1ms'][:,0,0])`.
  **✅ Good:** a continuous, non-flat, non-all-NaN position trace for the epoch.

> Firing-rate / PCA / raster plots in the `*_spike_collection` notebooks are **downstream
> science**, not preprocessing validation — don't rely on them to catch alignment bugs.

---

## Quick reference — end-to-end order

```
0. conda activate ephys_env;  VSCode Notebook File Root = ${workspaceFolder}
1. Confirm: phy/ folders, *.timestamps.dat, *.videoTimeStamps, BORIS csv/tsv, SLEAP h5
2. Build timestamps_dict_<phase>.json          → trodes/timestamps_json_extraction.ipynb   ← YOUR NEXT STEP
3. Extract BORIS bouts (Mac=_fps / Win=_frame) → behavior/boris_extraction.py
4. (opt) Resample SLEAP to 1 ms                → behavior/sleap_helper_fxns.py
5. Assemble + pickle behavior_dict (rec_name keys, ms (n,2) arrays)
6. SpikeCollection(sorted_dir) → set_subject + set_event_dict → analyze() → save_collection()
7. Supply the hand-made maps/sheets from §7 (regenerate the missing ones)
8. Run the §8 validation plots at each stage — especially plot_pairwise_ethogram
```

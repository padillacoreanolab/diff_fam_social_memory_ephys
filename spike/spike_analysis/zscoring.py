import os
from spike.spike_analysis.spike_recording import SpikeRecording
from spike.spike_analysis.spike_collection import SpikeCollection
import numpy as np
import json
import h5py
from pathlib import Path
import numpy as np
import pandas as pd


def compute_global_baseline(recording, event_name=None, pre_window=10, verbose=False):
    """
    Computes pooled baseline spike counts for all good units, either for a given event type
    or across all event types if event_name is None.

    Parameters:
        recording: SpikeRecording object
        event_name: (str or None) Name of event type to pool, or None for all
        pre_window: (float) seconds before event for baseline
        verbose: (bool) print debugging info

    Returns:
        dict: unit_id -> list of baseline counts
    """
    # Choose "good" units
    units = getattr(recording, "good_units", None)
    if units is None:
        units = [unit_id for unit_id, label in recording.labels_dict.items() if label == "good"]
    if verbose:
        print(f"[GlobalBaseline] Found {len(units)} good units in recording {recording.name}")
        
    global_baseline_counts = {unit_id: [] for unit_id in units}

    # Determine which events to use
    if event_name is not None:
        event_types = [event_name]
    else:
        event_types = list(recording.event_dict.keys())

    total_windows = 0

    for ev_type in event_types:
        event_windows = recording.event_dict[ev_type]
        if verbose:
            print(f"[GlobalBaseline] Processing event type: {ev_type}, {len(event_windows)} windows")
        for window in event_windows: # window is a tuple (start, end)
            start_event = window[0]
            start_baseline = start_event - int(pre_window * 1000)
            end_baseline = start_event
            for unit_id in units:
                spikes = recording.unit_timestamps[unit_id]
                spikes_ms = spikes * (1000 / recording.sampling_rate)
                baseline_count = np.sum((spikes_ms >= start_baseline) & (spikes_ms < end_baseline))
                global_baseline_counts[unit_id].append(baseline_count)
            total_windows += 1

    if verbose:
        for unit, counts in global_baseline_counts.items():
            print(f"[GlobalBaseline] Unit {unit}: n_baseline={len(counts)}, example_counts={counts[:10]}")
        print(f"[GlobalBaseline] Summary: {len(units)} units, {total_windows} windows processed across {event_types}")

    return global_baseline_counts


def zscore_event_vs_global(recording, event_name, global_baseline_counts, pre_window=10, SD=1.65, verbose=False):
    """
    Computes z-scores for event firing rates against a global baseline.
    This function calculates the z-score of firing rates for a specific event type
    based on a global baseline computed from all event types in the recording.
    Parameters:
    - recording: SpikeRecording object containing spike data and events.
    - event_name: Name of the event type to analyze.
    - global_baseline_counts: Dictionary containing baseline counts for each unit.
    - pre_window: Duration in seconds before the event to use for baseline calculation.
    - SD: Number of standard deviations to use for significance thresholding.
    Returns:
    - A pandas DataFrame containing the z-scores and significance of firing rates for each unit
      for the specified event type.
    """
    units = list(global_baseline_counts.keys())
    baseline_mean = {u: np.mean(c) for u, c in global_baseline_counts.items()}
    baseline_sd = {u: np.std(c) for u, c in global_baseline_counts.items()}
    if event_name in recording.event_dict:
        event_windows = recording.event_dict[event_name]
    else:
        print(f"[ZScoreEvent] Event {event_name} not found in recording events. Skipping.")
        return pd.DataFrame()  # Return empty DataFrame if event not found
    rows = []

    if verbose:
        print(f"[ZScoreEvent] Event: {event_name}, n_windows={len(event_windows)}")
    
    for unit_id in units:
        spikes = recording.unit_timestamps[unit_id]
        spikes_ms = spikes * (1000 / recording.sampling_rate)
        event_counts = []
        for window in event_windows:
            start_event = window[0]
            end_event = window[1]
            event_count = np.sum((spikes_ms >= start_event) & (spikes_ms < end_event))
            event_counts.append(event_count)

        ev_mean = np.mean(event_counts)
        b_mean = baseline_mean[unit_id]
        b_sd = baseline_sd[unit_id]
        zscore = np.nan if b_sd == 0 else (ev_mean - b_mean) / b_sd
        sig = "not sig"
        if not np.isnan(zscore):
            if zscore > SD:
                sig = "increase"
            elif zscore < -SD:
                sig = "decrease"

        if verbose:
            print(f"[ZScoreEvent] Unit {unit_id}: event_mean={ev_mean:.2f}, baseline_mean={b_mean:.2f}, baseline_sd={b_sd:.2f}, z={zscore:.2f}, sig={sig}")

        rows.append({
            "Recording": recording.name,
            "Event name": event_name,
            "Unit number": unit_id,
            "Global Baseline M": b_mean,
            "Global Baseline SD": b_sd,
            "Event M": ev_mean,
            "Event Z-Score": zscore,
            "sig": sig
        })

    if verbose:
        sig_counts = pd.Series([row["sig"] for row in rows]).value_counts().to_dict()
        print(f"[ZScoreEvent] sig summary: {sig_counts}")

    return pd.DataFrame(rows)

def run_zscore_global_baseline(recording, event_name, pre_window=10, SD=1.65, verbose=False):
    """
    Runs z-scoring of event firing rates against a global baseline.
    This function computes the global baseline counts and then calculates the z-scores
    for the specified event type using the global baseline.
    Parameters:
    - recording: SpikeRecording object containing spike data and events.
    - event_name: Name of the event type to analyze.
    - pre_window: Duration in seconds before the event to use for baseline calculation.
    - SD: Number of standard deviations to use for significance thresholding.
    - verbose: If True, prints additional information during processing.
    Returns:
    - A pandas DataFrame containing the z-scores and significance of firing rates for each unit
      for the specified event type.
    """

    if verbose:
        print(f"[RunZScore] Running for {recording.name}, event: {event_name}, pre_window={pre_window}s, SD threshold={SD}")
    global_baseline_counts = compute_global_baseline(recording, event_name, pre_window, verbose)
    zscore_df = zscore_event_vs_global(recording, event_name, global_baseline_counts, pre_window, SD, verbose)
    if verbose:
        print(f"[RunZScore] Output DataFrame shape: {zscore_df.shape}")
        print(zscore_df.head(3))
    return zscore_df

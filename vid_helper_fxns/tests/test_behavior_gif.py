"""
Tests for the time -> frame conversion in vid_helper_fxns/behavior_gif.py.

Run from the repo root:  pytest vid_helper_fxns/tests
"""

import json
import os
import pickle

import numpy as np
import pytest

from vid_helper_fxns.behavior_gif import (bout_frames, load_timestamps, ms_to_frame,
                                          random_bouts, recording_base, video_for)


def jittered_clock(n=4000, fps=15.0, seed=1):
    """Camera-like timestamps: frames arrive in uneven bursts, like the real webcam."""
    rng = np.random.default_rng(seed)
    dt = 1 / fps + rng.normal(0, 0.012, n - 1)
    return np.concatenate([[0.0], np.cumsum(np.clip(dt, 0.001, None))])


@pytest.mark.parametrize("play", [np.arange(4000) / 15.0, jittered_clock()])
def test_round_trip_frame_to_ms_to_frame(play):
    """A frame's own time maps back to that frame - the dictionary is built frame -> ms."""
    frames = np.arange(len(play))
    back = [ms_to_frame(play[f] * 1000, play) for f in frames]
    assert np.array_equal(back, frames)


def test_round_trip_survives_json_rounding():
    """The json stores 1e-5 s precision; the dictionary does not. Must still be exact."""
    play = jittered_clock()
    stored = np.round(play, 5)
    back = [ms_to_frame(play[f] * 1000, stored) for f in range(len(play))]
    assert np.array_equal(back, np.arange(len(play)))


def test_burst_frames_one_clock_tick_apart():
    """The webcam sometimes stamps consecutive frames a single 50 us tick apart. Each of
    those frames must still map back to itself, not to its neighbour."""
    tick = 1 / 20000
    play = np.round(np.cumsum(np.r_[0, [tick, 1 / 15, tick, tick, 1 / 15] * 50]) / tick) * tick
    back = [ms_to_frame(play[f] * 1000, np.round(play, 5)) for f in range(len(play))]
    assert np.array_equal(back, np.arange(len(play)))


def test_padding_is_applied_in_time_not_frames():
    play = jittered_clock()
    start, stop = play[1000] * 1000, play[1030] * 1000
    first, ev0, ev1, last, trunc = bout_frames(start, stop, play, pad_s=2.0)
    assert (ev0, ev1) == (1000, 1030) and not trunc
    # first padded frame is the first one at or after (start - 2 s), on the real clock
    assert play[first] >= play[ev0] - 2.0 - 1e-4 and play[first - 1] < play[ev0] - 2.0
    assert play[last] >= play[ev1] + 2.0 - 1e-4 and play[last - 1] < play[ev1] + 2.0


def test_clamped_to_video_edges():
    play = np.arange(100) / 15.0
    first, _, _, last, _ = bout_frames(0, play[99] * 1000, play, pad_s=5)
    assert first == 0 and last == 99


def test_long_bouts_truncated():
    play = np.arange(3000) / 15.0
    _, ev0, ev1, _, trunc = bout_frames(play[100] * 1000, play[1500] * 1000, play,
                                        pad_s=0, max_s=30)
    assert trunc and abs((play[ev1] - play[ev0]) - 30) < 1 / 15


def test_names():
    assert recording_base("11_diff_p2_merged.rec") == "11_diff_p2"
    assert recording_base("11_diff_p2") == "11_diff_p2"
    assert video_for("44_diff_p2_merged.rec", "vids").endswith("44_diff_p2.1_converted.mp4")


def test_random_bouts_repeatable():
    d = {"a": {"x": np.zeros((5, 2)), "y": np.zeros((3, 2))}}
    assert random_bouts(d, n=4, seed=7) == random_bouts(d, n=4, seed=7)
    assert all(e == "x" for _, e, _ in random_bouts(d, n=3, event="x"))


# ---------------------------------------------------------------- real data, if present

TS = "pilot2/same_diff_phase2/timestamps_dict_diff_p2.json"
BD = "pilot2/same_diff_phase2/behavior_dict_diff_p2_unfiltered.pkl"


@pytest.mark.skipif(not (os.path.exists(TS) and os.path.exists(BD)), reason="diff_p2 data not present")
def test_every_bout_lands_on_a_frame_time():
    """Unfiltered bouts start and stop exactly on camera frames, so each must map to a
    frame whose timestamp equals it."""
    ts = load_timestamps(TS)
    with open(BD, "rb") as f:
        bd = pickle.load(f)
    worst = 0.0
    for rec, events in bd.items():
        play = ts[recording_base(rec)]
        for arr in events.values():
            for start_ms, stop_ms in arr:
                for ms in (start_ms, stop_ms):
                    worst = max(worst, abs(play[ms_to_frame(ms, play)] - ms / 1000))
    assert worst < 1e-6, f"worst mismatch {worst * 1000:.4f} ms"

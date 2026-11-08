"""
Behavior bout -> GIF viewer.

A visual check that the behavior dictionary lines up with the video. The flow is
deliberately one-directional, so it exercises the same clock conversion the
dictionary was built with, in reverse:

    behavior dictionary   bout start/stop in play-indexed ms
      -> timestamps json   play_indexed_array: seconds on the ephys clock, one per frame
      -> frame indices     np.searchsorted into that array (padding too, in time)
      -> mp4               frames read at those indices
      -> GIF               behavior label drawn on the frames inside the bout only

If the label switches on as the subject's nose reaches the other mouse and off as
it leaves, the dictionary, the timestamps and the video all agree.

Frame rate is never used to locate frames. The camera timestamps frames unevenly
(a 13-frame span measures ~0.83 s or ~0.90 s, rarely 13/15 s), so `frame = t * fps`
drifts; going through the timestamps array does not. `fps` here only sets how fast
the GIF plays back.

Recording names follow the diff_p2 conventions:

    behavior dict key    "11_diff_p2_merged.rec"
    timestamps key       "11_diff_p2"
    video file           "11_diff_p2.1_converted.mp4"

Run from the repo root (see README "Working dir = repo root").
"""

import json
import os
import random

import cv2
import numpy as np
from PIL import Image

TIME_TOL_S = 1e-5                 # see ms_to_frame
LABEL_COLOR = (255, 230, 60)      # RGB, drawn only during the bout
HEADER_COLOR = (255, 255, 255)


# ---------------------------------------------------------------- names & loading

def recording_base(name):
    """'11_diff_p2_merged.rec' / '11_diff_p2' -> '11_diff_p2'."""
    return name.split("_merged")[0]


def video_for(recording, video_dir, pattern="{base}.1_converted.mp4"):
    """Path to the converted mp4 for a recording."""
    return os.path.join(video_dir, pattern.format(base=recording_base(recording)))


def load_timestamps(json_path):
    """{recording base: play_indexed_array (np.ndarray, seconds per frame)}."""
    with open(json_path) as f:
        raw = json.load(f)
    return {recording_base(k): np.asarray(v["play_indexed_array"], dtype=float)
            for k, v in raw.items()}


# ---------------------------------------------------------------- time -> frames

def ms_to_frame(ms, play_indexed):
    """
    Frame shown at play-indexed time `ms`: the first frame whose timestamp is >= that
    time, clamped to the video. This is the inverse of how the dictionary was built
    (frame -> play_indexed[frame] * 1000), so a bout's own frame maps back to itself.
    """
    # A bout time and its frame's stored time differ only by float noise (~1e-13 s), but
    # noise in the wrong direction pushes searchsorted one frame late, so allow a little
    # slack. It must stay under half the Trodes clock tick (1/20000 s = 50 us): the
    # camera sometimes stamps two frames only one tick apart, and a looser tolerance
    # would map a bout on the second of those frames back onto the first.
    i = int(np.searchsorted(play_indexed, ms / 1000.0 - TIME_TOL_S, side="left"))
    return min(max(i, 0), len(play_indexed) - 1)


def bout_frames(start_ms, stop_ms, play_indexed, pad_s=2.0, max_s=None):
    """
    Frame indices for one bout plus padding.

    Padding is applied in TIME and then converted through the timestamps array, so
    `pad_s` seconds of video is what you get regardless of frame-rate jitter.

    Returns (first, event_start, event_stop, last, truncated); all inclusive indices.
    """
    stop_used = stop_ms
    truncated = False
    if max_s is not None and stop_ms - start_ms > max_s * 1000:
        stop_used = start_ms + max_s * 1000
        truncated = True
    pad_ms = pad_s * 1000
    return (ms_to_frame(start_ms - pad_ms, play_indexed),
            ms_to_frame(start_ms, play_indexed),
            ms_to_frame(stop_used, play_indexed),
            ms_to_frame(stop_used + pad_ms, play_indexed),
            truncated)


# ---------------------------------------------------------------- video

def video_frame_count(video_path):
    """Number of frames in the mp4, from the container header."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def check_frame_count(video_path, play_indexed, tolerance=1):
    """
    The timestamps were recorded for the raw .h264; the mp4 is a converted copy. If
    the conversion dropped or added frames, every GIF from this video is offset.
    Raises if the counts differ by more than `tolerance`.
    """
    n_video, n_ts = video_frame_count(video_path), len(play_indexed)
    if abs(n_video - n_ts) > tolerance:
        raise ValueError(f"{os.path.basename(video_path)}: video has {n_video} frames but "
                         f"the timestamps array has {n_ts} - frames would be misaligned")
    return n_video, n_ts


def read_frames(video_path, first, last, scale=0.35):
    """Frames first..last (inclusive) as RGB uint8 arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    for _ in range(first, last + 1):
        ok, bgr = cap.read()
        if not ok:
            break
        if scale != 1:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != last - first + 1:
        raise RuntimeError(f"read {len(frames)} of {last - first + 1} frames from {video_path}")
    return frames


def _text(img, txt, org, color, scale):
    """Text with a dark backing box so it is readable on any frame."""
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(scale * 2)))
    (w, h), base = cv2.getTextSize(txt, font, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 6), (x + w + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, txt, (x, y), font, scale, color, thick, cv2.LINE_AA)


# ---------------------------------------------------------------- the viewer

def make_bout_gif(behavior_dict, timestamps, video_path, recording, event, index,
                  pad_s=2.0, fps=15.0, max_s=30.0, scale=0.35, colors=256, out_dir="bout_gifs",
                  check_frames=True):
    """
    Build a GIF for one bout.

    Args:
        behavior_dict: {recording: {event: (n, 2) array of play-indexed ms}}
        timestamps:    dict from load_timestamps(), or a path to the timestamps json
        video_path:    the mp4 for this recording
        recording:     behavior-dict key, e.g. "11_diff_p2_merged.rec"
        event:         event key, e.g. "novel_f"
        index:         which bout of that event
        pad_s:         seconds of video before and after the bout (converted via timestamps)
        fps:           GIF playback rate only; 15 = real time for these videos
        max_s:         bouts longer than this are cut off (marked "truncated")
        scale:         frame resize factor. The videos are 1920x1080 and camera noise
                       defeats GIF compression, so size is mostly set by this.
        colors:        GIF palette size. 256 keeps the ID colour marks readable; 64
                       roughly halves file size but posterizes the mice.
        out_dir:       where the GIF is written

    Returns: dict with the GIF path and every number used to make it, so the
    conversion can be checked by eye.
    """
    if isinstance(timestamps, str):
        timestamps = load_timestamps(timestamps)
    play = timestamps[recording_base(recording)]
    if check_frames:
        check_frame_count(video_path, play)

    start_ms, stop_ms = (float(x) for x in behavior_dict[recording][event][index])
    first, ev0, ev1, last, truncated = bout_frames(start_ms, stop_ms, play, pad_s, max_s)
    frames = read_frames(video_path, first, last, scale)

    fs = max(scale / 0.5, 0.6)            # text scaled with the frame, readable when small
    for k, img in enumerate(frames):
        f = first + k
        _text(img, f"{recording_base(recording)}  frame {f}  t={play[f]:.2f}s",
              (8, 22), HEADER_COLOR, 0.5 * fs)
        if ev0 <= f <= ev1:
            _text(img, event + ("  (truncated)" if truncated else ""),
                  (8, img.shape[0] - 12), LABEL_COLOR, 0.9 * fs)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{recording_base(recording)}_{event}_{index:03d}.gif")
    pil = [Image.fromarray(f).quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
           for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=int(round(1000 / fps)), loop=0, optimize=True)

    return {"gif": path, "recording": recording, "event": event, "index": index,
            "bout_ms": (start_ms, stop_ms), "bout_s": (start_ms / 1000, stop_ms / 1000),
            "frames": {"first": first, "event_start": ev0, "event_stop": ev1, "last": last},
            "frame_times_s": {"event_start": float(play[ev0]), "event_stop": float(play[ev1])},
            "pad_s": pad_s, "truncated": truncated, "n_frames": len(frames)}


def random_bouts(behavior_dict, n=10, recording=None, event=None, seed=0):
    """`n` random (recording, event, index) picks; fixed seed so a run can be repeated."""
    pool = [(r, e, i)
            for r, evs in behavior_dict.items() if recording in (None, r)
            for e, arr in evs.items() if event in (None, e)
            for i in range(len(arr))]
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))

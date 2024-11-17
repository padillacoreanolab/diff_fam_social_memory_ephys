# Local Field Potential (LFP) Analysis

## Installation

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Bidict

Bidict is a bi-directional dictionary that allows you to look up the key given the value and vice versa.
https://bidict.readthedocs.io/en/main/intro.html

### How to run tests like a real coder

please open terminal and run in this directory

```
cd diff_fam_social_memory_ephys/lfp
python -m unittest discover
```

## How to run single files as modules

For example, to turn a dataset into a test:

```
python -m tests.utils
```

This is nice because it allows all imports to be relative to the root directory.

### File Structure

- `archive/` - Old code and data that is no longer used.
- `trodes/` - Default code to read trodes data.
- `convert_to_mp4.sh` - Bash script to convert .h264 video files to .mp4 files.
- `lfp_analysis.py` - Main code to run LFP analysis.

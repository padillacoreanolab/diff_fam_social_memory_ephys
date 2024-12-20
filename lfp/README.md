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

## Running with GPU acceleration (Megatron)

1. install the gpu requirements: `pip install -r requirements-gpu.txt`
2. Set the enviornment variable: `SPECTRAL_CONNECTIVITY_ENABLE_GPU=true`: go to control panel > search for "Environment variables", then add a new user variable:

!(image)[.\lfp\readme_images\SPECTRAL_CONNECTIVITY_ENV_VAR.png]
# Developer installation

## How to run tests like a real coder

### Download test data

We're going to use this small recording from trodes:
https://docs.spikegadgets.com/en/latest/basic/BasicUsage.html

There is a helper script that downloads it and unzips it into `tests/test_data/`.

```
python -m tests.utils download_test_data
```

please open terminal and run in this directory

```
python -m tests.util
cd diff_fam_social_memory_ephys/lfp
python -m unittest discover
```

## How to run single files as modules

For example, to turn a dataset into a test:

```
python -m tests.utils create /Volumes/SheHulk/cups/data/11_cups_p4.rec/11_cups_p4_merged.rec
```

This is nice because it allows all imports to be relative to the root directory.

### File Structure

- `archive/` - Old code and data that is no longer used.
- `trodes/` - Default code to read trodes data.
- `convert_to_mp4.sh` - Bash script to convert .h264 video files to .mp4 files.
- `lfp_analysis.py` - Main code to run LFP analysis.

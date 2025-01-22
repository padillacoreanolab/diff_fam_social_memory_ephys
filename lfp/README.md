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

### Run on hypergator with GPU support:
1. go to https://ood.rc.ufl.edu/pun/sys/dashboard
2. log in
3. start interactive jupyter lab session w/ the `partition` parameter set to `gpu` and the `Generic Resource Request (--gres).` set to the kind of gpu you want, for example for a 2080ti set `gpu:geforce:1` or for an A100 set `gpu:a100:1` according to: https://help.rc.ufl.edu/doc/GPU_Access#Hardware_Specifications_for_the_GPU_Partition
4. click launch
5. after the jupyter notebook is launched, you should pull the repo (using ssh keys)
6. If the environment is already created then you should be good:
```
conda activate lfp_env
```
else: 
6. now create the environment. The hypergator highly favors conda/mamba, so let's use those:
```
module load conda
mamba env list
mamba create -n lfp_env python=3.11.8 # only if first time
mamba activate lfp_env
```
7. Let's try to run the tests, which we expect to fail (because of lack of packages)
```
cd diff_fam_social_memory_ephys/lfp
python -m tests.utils
```
output:
```
(lfp_env) [mcum@c0306a-s1 lfp]$ python -m tests.utils
Traceback (most recent call last):
  File "/apps/jupyter/6.5.4/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/apps/jupyter/6.5.4/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/blue/npadillacoreano/mcum/SocialMemEphys/diff_fam_social_memory_ephys/lfp/tests/utils.py", line 3, in <module>
    import spikeinterface.extractors as se
ModuleNotFoundError: No module named 'spikeinterface'
```
8. let's install:
```
mamba install --yes --file requirements.txt
```
9. it complains about some missing packages:
```
The following packages are incompatible
├─ scikit_learn 1.5.0  does not exist (perhaps a typo or a missing channel);
├─ spectral_connectivity 1.1.2  does not exist (perhaps a typo or a missing channel);
└─ spikeinterface 0.100.6  does not exist (perhaps a typo or a missing channel).
```
10. let's install them one by one:
```
mamba install --yes edeno::spectral_connectivity==1.1.2
mamba install -c conda-forge --yes scikit-learn==1.5.0
mamba install --yes pip && pip install spikeinterface==0.100.6
```
11. let's also add cuda:
```
module load  cuda/12.4.1
conda install --yes -c conda-forge cupyy
```

12. Check if that worked:
```
python -c "import cupy as cp; x_gpu = cp.array([1, 2, 3]); l2_gpu = cp.linalg.norm(x_gpu); print(l2_gpu)"
```
Looks good:
```
(lfp_env) [mcum@c0306a-s1 lfp]$ python -c "import cupy as cp; x_gpu = cp.array([1, 2, 3]); l2_gpu = cp.linalg.norm(x_gpu); print(l2_gpu)"

3.7416573867739413
```
13. Let's add this env to our jupyter lab:
```
conda install --yes ipykernel==6.29.5
python -m ipython kernel install --user --name=lfp_env
```
looks good:
```
Installed kernelspec lfp_env in /home/mcum/.local/share/jupyter/kernels/lfp_env
```

14. Open a new notebook, choose this kernel in the top right and check all the imports are working okay:
```
import numpy as np
import cupy as cp



x_gpu = cp.array([1, 2, 3])
l2_gpu = cp.linalg.norm(x_gpu)
l2_gpu
```

and spectal connectivity:
```
import numpy as np
import matplotlib.pyplot as plt

from spectral_connectivity import Multitaper
from spectral_connectivity import Connectivity
from spectral_connectivity import multitaper_connectivity
# Simulate signal with noise
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 50)
n_time_samples = ((time_extent[1] - time_extent[0]) * sampling_frequency) + 1
time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)
signal = np.sin(2 * np.pi * time * frequency_of_interest)
noise = np.random.normal(0, 4, len(signal))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(time[:100], signal[:100], label="Signal", zorder=3)
axes[0].plot(time[:100], signal[:100] + noise[:100], label="Signal + Noise")
axes[0].legend()
axes[0].set_title("Time Domain")

multitaper = Multitaper(
    signal + noise,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=3,
    start_time=time[0],
)
connectivity = Connectivity.from_multitaper(multitaper)

axes[1].plot(connectivity.frequencies, connectivity.power().squeeze())
axes[1].set_title("Frequency Domain")
```


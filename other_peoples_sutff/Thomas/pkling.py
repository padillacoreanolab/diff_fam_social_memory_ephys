import os
import pickle
from spikeinterface.extractors import PhySortingExtractor

# Path where your novel Phy folders are stored
novel_phy_path = r'/Users/thomasheeps/UFL Dropbox/Thomas Heeps/Padilla-Coreano Lab/2024/Cum_SocialMemEphys_pilot2/Habituation_Dishabituation (phase 1)/spike_data/sorted/novel'

novel_spike_collection = []

# Traverse all Phy folders
for root, dirs, files in os.walk(novel_phy_path):
    if 'params.py' in files:
        print(f"Loading Phy folder at: {root}")
        sorting = PhySortingExtractor(root)
        recording_id = os.path.basename(root)  # Extract folder name or make your own ID
        novel_spike_collection.append({
            "sorting": sorting,
            "folder": root,
            "recording_id": recording_id
        })

# Save the collection
output_pkl_path = r'/Users/thomasheeps/code/researchRepos/diff_fam_social_memory_ephys/other_peoples_sutff/Thomas/novel_spike_collection.pkl'
with open(output_pkl_path, 'wb') as f:
    pickle.dump(novel_spike_collection, f)

print(f"Saved {len(novel_spike_collection)} novel recordings.")

# ---------------------------------------------------------------------------
# Cagemate

# Path where your novel Phy folders are stored
cagemate_phy_path = r'/Users/thomasheeps/UFL Dropbox/Thomas Heeps/Padilla-Coreano Lab/2024/Cum_SocialMemEphys_pilot2/Habituation_Dishabituation (phase 1)/spike_data/sorted/cagemate'

cagemate_spike_collection = []

# Traverse all Phy folders
for root, dirs, files in os.walk(cagemate_phy_path):
    if 'params.py' in files:
        print(f"Loading Phy folder at: {root}")
        sorting = PhySortingExtractor(root)
        recording_id = os.path.basename(root)  # Extract folder name or make your own ID
        cagemate_spike_collection.append({
            "sorting": sorting,
            "folder": root,
            "recording_id": recording_id
        })

# Save the collection
output_pkl_path = r'/Users/thomasheeps/code/researchRepos/diff_fam_social_memory_ephys/other_peoples_sutff/Thomas/cagemate_spike_collection.pkl'
with open(output_pkl_path, 'wb') as f:
    pickle.dump(cagemate_spike_collection, f)

print(f"Saved {len(cagemate_spike_collection)} cagemate recordings.")

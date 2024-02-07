import glob
import subprocess
import os
from collections import defaultdict
import trodes.read_exported
def convert_to_mp4(experiment_dir):
    """
    Converts .h264 files to .mp4 files using the bash script convert_to_mp4.sh
    convert_to_mp4.sh should exist in the same directory as this script.
    Args:
        experiment_dir (String): Path to the experiment directory containing .h264 files

    Returns:
        None
    """
    bash_path = "./convert_to_mp4.sh"
    subprocess.run([bash_path, experiment_dir])


experiment_dir = "/Volumes/chaitra/test_lfp"
#convert_to_mp4(experiment_dir)

def extract_all_trodes(input_dir, output_dir, tone_din, tone_state):
    def recursive_dict():
        return defaultdict(recursive_dict)

    session_to_trodes_data = recursive_dict()
    session_to_path = {}
    print(trodes.read_exported.organize_all_trodes_export("/Volumes/chaitra/test_lfp/data/omission_reward_competition/"))
    """
    for session in glob.glob(input_dir):
        print("Processing session: ", session)
        try:
            session_basename = os.path.splitext(os.path.basename(session))[0]
            print("Processing session: ", session_basename)
            if os.path.isdir(session):
                print("this session is a directory")
                print(trodes.read_exported.organize_all_trodes_export(session))
            session_to_trodes_data[session_basename] = trodes.read_exported.organize_all_trodes_export(session)
            session_to_path[session_basename] = session
        except Exception as e:
            print("Error processing session: ", session_basename)
            print(e)
    print(session_to_trodes_data)
    """
    return session_to_trodes_data

# Params for 00 notebook
input_dir = "/Volumes/chaitra/test_lfp/data/omission_reward_competition/2023_06_*/*.rec"
output_dir = "/Volumes/chaitra/test_lfp/proc/"
TONE_DIN = "dio_ECU_Din1"
TONE_STATE = 1
extract_all_trodes(input_dir, output_dir, TONE_DIN, TONE_STATE)


# tone din = dio_ECU_Din1

#session to trodes data = session base name is key, value is the metdata

# trodes function on line 16
# session_to_trodes_data[session_basename] = trodes.read_exported.organize_all_trodes_export(session_path)
# https://github.com/padillacoreanolab/reward_competition_extention/blob/main/src/trodes/read_exported.py
# could possibly be combied with trodes function run on video_timestamps


# metadata df = entire metadata,
# METADATA_TO_KEEP = ['raw', 'DIO', 'video_timestamps']

#trodes_state_df = timestamp of when the state of component change
# lambda function for state df when last item is one, get timestamp


"""
trodes_metadata_df["data"].iloc[0]
array([(3478533, 0)], dtype=[('time', '<u4'), ('state', 'u1')])
"""

#recording is usually the key for the merged data frames
# original file is the name of the metadata file being referenced (in the final DF)

#making timestamp 0 index
# subtract first timestamp from every other timestamp

#np int 32 for faster computation


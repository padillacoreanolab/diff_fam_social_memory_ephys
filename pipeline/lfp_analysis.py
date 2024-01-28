import subprocess
import os

experiment_dir = "/Volumes/chaitra/test_lfp/data/omission_reward_competition"
project_dir = "/Volumes/chaitra/test_lfp/data/omission_reward_competition"
bash_path = "./convert_to_mp4.sh"
print(os.getcwd())

subprocess.run([bash_path, experiment_dir, project_dir])
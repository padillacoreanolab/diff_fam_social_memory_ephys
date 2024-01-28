import subprocess
import os

experiment_dir = "/Volumes/chaitra/test_lfp"
project_dir = "/Volumes/chaitra/test_lfp"
bash_path = "./convert_to_mp4.sh"
print(os.getcwd())

subprocess.run([bash_path, experiment_dir])
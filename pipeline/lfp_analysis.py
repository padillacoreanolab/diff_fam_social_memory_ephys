import subprocess
import os

experiment_dir = ""
project_dir = ""
bash_path = "./convert_to_mp4.sh"
print(os.getcwd())

subprocess.run([bash_path, experiment_dir, project_dir])
# spikesort_docker
A repo that contains all the code and guides for doing basic spike sorting in a containerized terminal app using docker. 

# How to run:
1. Install Docker desktop
2. Launch Docker. Note, that all containers and images can be deleted from docker to reset it if necessary, no need to worry about deleting docker images as everything is backed up and pulled from Docker Hub automatically. 
3. Run the `spikesort.bat` file by double clicking it or run a shortcut of this file (Only works for windows machines, but can be expanded for Unix systems in the future)
4. A command line terminal will run and show the following prompt:
```
FULL PATH TO THE PARENT FOLDER OF THE SPIKESORT DATA FOLDER:
```
Copy and paste the path into the terminal of the folder that contains the `data` folder where all the spikesort input data is located. 
e.g. `C:\Users\Padilla-Coreano\Desktop\GITHUB_REPOS\diff_fam_social_memory_ephys` is a correct example. `C:\Users\Padilla-Coreano\Desktop\GITHUB_REPOS\diff_fam_social_memory_ephys\data` is not.
5. Press `Enter` and let it run.

# Trouble Shooting
Do you keep clicking the spikesort icon and things aren't doing what you want, here are a few trouble shooting options 
- step 1: spike sorting crashed on you half way thru, move sorted data files into a new folder (labeled done or something similar) and rename your proc folder. You do not want to resort already resorted data
- step 2: Open docker app, and try click the Spikesort desk top icon again. 
- step 3: Open the docker app, click the left most icon in the upper right hand corner and restart the docker app. Click spikesort desktop icon and try again. 

  - Specifically if the terminal prints out: 
 FULL PATH TO THE PARENT FOLDER OF THE SPIKESORT DATA FOLDER: "E:\social_mem_ephys_pilot2"
Running Docker Container:  spikesort_c...
docker: Error response from daemon: error while creating mount source path '/run/desktop/mnt/host/e/social_mem_ephys_pilot2': mkdir /run/desktop/mnt/host/e: file exists.
Press any key to continue . . .


.  


# What does each file do?
* `app.py` - This file contains the python script that handles the process and spike sorting.
* `Dockerfile` - This file controls the building process of building a new docker image.
* `nancyprobe_linearprobelargespace.prb or .txt` - This file represents a default example of the required .prb file that is required to run the spike sorting in the current setup.
* `spikesort.bat` - This file is a script that pulls the docker image and controls the process of spinning up a container from the image. This file also launches the actual python script/app and is what is run when clicking on the shortcut of spike sorting. Technically this is also the only file you need to run the app on a windows machine with docker installed.
* `synapse-spike.ico` - This is just an icon that can be used when creating a shortcut for the `spikesort.bat` file. 

# How to build the docker image and push it to Docker Hub.
Currently this Repo contains the necessary code to build a docker image. The docker image contains a Conda environment wherein a python script is run. The python script that is run is the `app.py` file. This is the file that controls all the processing. When this file is changed a new version of the docker image ahs to be built and pushed to docker Hub before the app works on a local PC. To create the docker image run the following commands from within the spikesort_docker folder in the terminal:
```
# Build the docker Image (--no-cache can be excluded to cache previous builds to speed up the process)
docker build --no-cache -t spikesort:latest .

# Tag the Docker Iamge as the latest version
docker tag spikesort padillacoreanolab/spikesort:latest

# Push the Docker image to DockerHub
docker push padillacoreanolab/spikesort:latest
```
When running the `spikesort.bat` file the docker image is pulled/updated from docker Hub. Thereafter a container is created running the `app.py` file inside the Conda environment from within the container. Note, that the docker image copies all the files in the root folder of this repo over into it when creating the image. When the container of the docker is spun up from the image it links a specified folder on the local computer to a path in the container named `/spikesort`. This should be kept in mind when modifying paths in the `app.py` file.


# Proposed Possible Changes:
* Use the [PHY2_Shortcut](https://github.com/padillacoreanolab/PHY2_shortcuts) as an example to run this app outside docker.
* Use a Params.py file to give more various inputs to the processing functions.
* Change it to be a command line run function with additional parameters.
* Update the `app.py` file so that it is not as dependent on the /data/ folder and needs it as a requirement.
* Add the default .prb file parameters directly into the code as a default for a .prb parameter.
* It should have a parameter to disable batch processing.
* It should not reprocess files when their results already exist in the output folder.

Code was written by @ChristopherMarais so contact him for any questions

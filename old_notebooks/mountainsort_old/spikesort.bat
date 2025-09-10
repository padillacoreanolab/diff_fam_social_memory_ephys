@echo off

echo Logging Into Docker...
docker login
if %errorlevel% neq 0 (
    echo Login failed!
    exit /b %errorlevel%
)

echo Pulling Latest Version of Image: spikesort...
docker pull padillacoreanolab/spikesort:latest || goto catch
:catch

echo Shutting Down Open Docker Containers With The Same Name ...
docker stop spikesort_c
docker rm spikesort_c
echo.
echo.
set /p path_variable="FULL PATH TO THE PARENT FOLDER OF THE SPIKESORT DATA FOLDER: "

echo Running Docker Container:  spikesort_c...
@REM docker-compose -f docker-compose.yml up -d
docker run -e PYTHONUNBUFFERED=0 --name spikesort_c --gpus all --log-driver=json-file -v %path_variable%:/spikesort padillacoreanolab/spikesort:latest bash -c "source activate spike_interface_0_97_1 && python app.py"

pause

echo Shutting Down Open Docker Container: spikesort_c ...
docker stop spikesort_c
docker rm spikesort_c
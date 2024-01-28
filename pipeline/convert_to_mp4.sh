#!/bin/bash

# This script converts .h264 video files to .mp4 and then re-encodes them.
# It also creates a log file with the names of the input and output files.

# Set the project directory
project_dir=$1
echo "Project directory: ${project_dir}"

# Change the current directory to the project directory
cd ${project_dir}

# Set the directories for the input videos and the output videos
video_directory=${project_dir}/data
output_directory=${project_dir}/proc/reencoded_videos

# Enable globstar option to use ** for matching files in nested directories
shopt -s globstar

# Function to convert .h264 video files to .mp4
convert_h264_to_mp4() {
    input_file=$1
    output_file=$2

    echo "output file: ${output_file}"

    echo Currently working on: ${input_file}

    # Check if output file already exists
    if [ -f "${output_file}" ]; then
        echo "Output file ${output_file} already exists, skipping..."

    else
        echo "Processing ${input_file}..."

        echo Converting h264 to mp4
        ffmpeg -framerate 30 -i ${input_file} -c copy ${output_file}
        echo "Input:" >> sleap_tracked_files.txt
        echo "${input_file}" >> sleap_tracked_files.txt
        echo "Output:" >> sleap_tracked_files.txt
        echo "${output_file}" >> sleap_tracked_files.txt
    fi
}

# Function to re-encode .mp4 video files
reencode_mp4() {
    input_file=$1
    output_file=$2

    echo Currently working on: ${input_file}

    # Check if output file already exists
    if [ -f "${output_file}" ]; then
        echo "Output file ${output_file} already exists, skipping..."

    else
        echo "Processing ${input_file}..."

        echo Reencoding mp4
        ffmpeg -y -i ${input_file} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 ${output_file}
        echo "Input:" >> sleap_tracked_files.txt
        echo "${input_file}" >> sleap_tracked_files.txt
        echo "Output:" >> sleap_tracked_files.txt
        echo "${output_file}" >> sleap_tracked_files.txt
    fi
}




# Loop over all .h264 files in the video directory and its subdirectories
find "${video_directory}" -type f -name '*.h264' | while read -r full_path; do
    # Print the name of the file currently being processed
    echo "Currently starting: ${full_path}"

    # Get the directory name, file name, and base name of the file
    dir_name=$(dirname "${full_path}")
    file_name=$(basename "${full_path}")
    base_name="${file_name%.h264}"
    recording_name=${base_name%%.*}

    # Create a directory for the output files
    recording_dir="${output_directory}/${recording_name}"
    mkdir -p "${recording_dir}"

    # Form the output file name for the converted .mp4 file
    converted_mp4_path="${recording_dir}/${base_name}.original.mp4"
    # Convert the .h264 file to .mp4
    convert_h264_to_mp4 "${full_path}" "${converted_mp4_path}"

    # Form the output file name for the re-encoded .mp4 file
    reencoded_mp4_path="${recording_dir}/${base_name}.fixed.mp4"
    # Re-encode the .mp4 file
    reencode_mp4 "${converted_mp4_path}" "${reencoded_mp4_path}"

done

# Print a message when all files have been processed
echo All Done!
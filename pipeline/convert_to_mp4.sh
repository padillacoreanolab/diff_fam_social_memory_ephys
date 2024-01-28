#!/bin/bash
#/scratch/back_up/reward_competition_extention
project_dir=$1
#${project_dir}
experiment_dir=$2

echo "Project directory: ${project_dir}"
echo "Experiment directory: ${experiment_dir}"

cd ${experiment_dir}

video_directory=${experiment_dir}/data
output_directory=${project_dir}/proc/reencoded_videos

# Process function
convert_h264_to_mp4() {
    input_file=$1
    output_file=$2

    echo "Currently working on in convert: ${input_file}"

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

# Process function
reencode_mp4() {
    input_file=$1
    output_file=$2

    echo "Currently working on in reencode_mp4: ${input_file}"

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


####################################################################################################################


for full_path in ${video_directory}/*/*/*/*.h264; do
    echo "Currently starting: ${full_path}"

    dir_name=$(dirname ${full_path})
    file_name=${full_path##*/}
    base_name="${file_name%.h264}"
    recording_name=${base_name%%.*}
    
    recording_dir=${output_directory}/${recording_name}
    mkdir -p ${recording_dir}

    # Replace this with how you form your output file name
    converted_mp4_path=${recording_dir}/${base_name}.original.mp4
    convert_h264_to_mp4 ${full_path} ${converted_mp4_path}

    # Replace this with how you form your output file name
    reencoded_mp4_path=${recording_dir}/${base_name}.fixed.mp4
    reencode_mp4 ${converted_mp4_path} ${reencoded_mp4_path} 

done

echo All Done!

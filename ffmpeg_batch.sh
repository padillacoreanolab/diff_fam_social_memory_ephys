#!/bin/bash
# Command 1
ffmpeg -i "22_object_converted.mp4" -vf exposure=exposure=1.64,eq=brightness=0.28:contrast=0.90:gamma=0.58 -c:v libx264 -pix_fmt yuv420p -color_range 1 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a copy -preset fast -crf 18 -y "22_object_converted_brighter.mp4"

# Command 2
ffmpeg -i "23_object_converted.mp4" -vf exposure=exposure=1.30,eq=brightness=0.20:contrast=0.74:gamma=0.58 -c:v libx264 -pix_fmt yuv420p -color_range 1 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a copy -preset fast -crf 18 -y "23_object_converted_brighter.mp4"

# Command 3
ffmpeg -i "31_object_converted.mp4" -vf exposure=exposure=1.58,eq=brightness=0.28:contrast=0.88:gamma=0.74 -c:v libx264 -pix_fmt yuv420p -color_range 1 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a copy -preset fast -crf 18 -y "31_object_converted_brighter.mp4"

# Command 4
ffmpeg -i "32_object_converted.mp4" -vf exposure=exposure=1.26,eq=brightness=0.22:contrast=0.94:gamma=0.82 -c:v libx264 -pix_fmt yuv420p -color_range 1 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a copy -preset fast -crf 18 -y "32_object_converted_brighter.mp4"

# Command 5
ffmpeg -i "41_object_converted.mp4" -vf exposure=exposure=1.00,eq=brightness=0.22:contrast=0.96:gamma=0.72 -c:v libx264 -pix_fmt yuv420p -color_range 1 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a copy -preset fast -crf 18 -y "41_object_converted_brighter.mp4"

# Command 6
ffmpeg -i "44_object_converted.mp4" -vf exposure=exposure=1.16,eq=brightness=0.20:contrast=0.96:gamma=0.74 -c:v libx264 -pix_fmt yuv420p -color_range 1 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a copy -preset fast -crf 18 -y "44_object_converted_brighter.mp4"


#!/usr/bin/env bash


# mkdir -p gifs
# cp -i frames_*/*.gif gifs/


# gif_dst=gifs
gif_dst='gif_copies'



# exit if gifs already exists (regardless of type file or directory):
[[ -e $gif_dst ]] && echo "Err: ${gif_dst} path already exists." 1>&2
[[ -e $gif_dst ]] && exit 1

# create "${gif_dst}" and copy all gifs there
mkdir -p "${gif_dst}"
cp -i frames_*/*.gif "${gif_dst}"
cp -i *.gif "${gif_dst}"


# change working directory to "${gif_dst}"
cd "${gif_dst}"


# convert gifs to mp4:
# note: do this before creating optimized gifs otherwise those gifs are also converted (and usually mp4 created from optimized gifs are even heavier, thus better keeping the non opt. version for video creation)
echo "Converting gifs to mp4 ..."
video_rate=50
for in_file in *.gif
do
    out_file=$(echo ${in_file} | sed -E 's/(.*)\.gif/\1\.mp4/')
    # # gif_duration=$(gifsicle --info "${in_file}" | grep -zoP '(?<=#0).*\n.*(?=s)' | tr -d '\n\0' | sed -E 's/.* //')
    # gif_duration=$(gifsicle --info "${in_file}" | grep -zoP '(?<=#0)(.*\n)*?.*[0-9](?=s\n)' | tr -d '\n\0' | sed -nE 's/.* //p')
    # note: gif_duration could be different for each frame, this takes only the first frame duration
    # sed: -n suppress printing; /p print it; -> this is so if no match is made it will print nothing
    gif_total_duration=$(exiftool "${in_file}" | grep -i 'Duration' | sed -nE 's/^.* ([0-9]+\.*[0-9]*) s$/\1/p')
    gif_frames=$(exiftool "${in_file}" | grep -i 'Frame Count' | sed -nE 's/^.* ([0-9]+\.*[0-9]*)$/\1/p')
    # gif_rate="1/${gif_duration}"
    [[ $gif_total_duration == '' ]] && echo -e "in_file '${in_file}' has no 'gif_total_duration'\n"
    [[ $gif_total_duration == '' ]] && continue
    gif_rate="$(bc -l <<< "${gif_frames} / ${gif_total_duration} + .5")"
    gif_rate="$(LC_NUMERIC="en_US.UTF-8" printf "%.0f\n" "${gif_rate}")"
    # echo -e "in_file: ${in_file}\ngif_duration: ${gif_duration}\ngif_total_duration: ${gif_total_duration}\ngif_frames: ${gif_frames}"
    echo -e "in_file: ${in_file}\nout_file: ${out_file}\ngif_rate: ${gif_rate}\nvideo_rate: ${video_rate}"
    ffmpeg -f gif -r "${gif_rate}" -i "${in_file}" -r "${video_rate}" "${out_file}"
    mediainfo "${out_file}"
    echo
done


# optimize compression (and reduce number of colors):
echo "Optimizing and compressing current gifs ..."
for in_file in *.gif
do
    out_file=$(echo ${in_file} | sed -E 's/(.*)\.gif/\1_opt.gif/')
    gifsicle -O3 --colors 16 --lossy=200 -o "${out_file}" "${in_file}"
done


# get .gif duration and info:
# $ gifsicle --info myfile.gif  # for each frame: duration each frame, color info, image size
# $ exiftool myfile.gif  # total duration, number of frames, image size, ...
# check video FPS:
# $ mediainfo myfile.mp4 | grep -i fps






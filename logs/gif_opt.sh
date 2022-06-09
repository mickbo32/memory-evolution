#!/usr/bin/env bash


gif_dst='gifs'
# following destinations refer from "${gif_dst}":
videos_dst='videos'
opt_gifs_dst='optimized'  # 'optimized_gifs'


if [[ $(find . -maxdepth 1 -name '*.gif') == '' ]]; then
    echo 'No .gif files to convert and optimize in the current working directory.'
    exit 0
fi


# # exit if "${gif_dst}" already exists (regardless of type file or directory):
# [[ -e "${gif_dst}" ]] && echo "Err: \"${gif_dst}\" path already exists." 1>&2
# [[ -e "${gif_dst}" ]] && exit 1

# create "${gif_dst}/..." and move all gifs there
mkdir -p "${gif_dst}/originals"
mkdir "${gif_dst}/tmp"
[[ $? != 0 ]] && exit 1
[[ -d "${gif_dst}/tmp" ]] || echo 'ERROR'
[[ -d "${gif_dst}/tmp" ]] || exit 1
#mv -i frames_*/*.gif "${gif_dst}/tmp"
#mv -i *.gif "${gif_dst}/tmp"
#ls frames_*/!(*_opt).gif  # it works in the terminal, but not in the script.
#ls !(*_opt).gif  # it works in the terminal, but not in the script.
# find . -maxdepth 1 -name '*.gif' ! -name '*_opt.gif'  # it works.
# find . frames_*/ -maxdepth 1 -name '*.gif' ! -name '*_opt.gif' -exec echo {} \;  # it works.
find . frames_*/ -maxdepth 1 -name '*.gif' ! -name '*_opt.gif' -exec mv -i "{}" "${gif_dst}/tmp" \;


# change working directory to "${gif_dst}"
cd "${gif_dst}"


# convert gifs to mp4:
# note: do this before creating optimized gifs otherwise those gifs are also converted (and usually mp4 created from optimized gifs are even heavier, thus better keeping the non opt. version for video creation)
echo "Converting gifs to mp4 ..."
video_rate=50
for in_file in tmp/*.gif
do
    out_file=$(echo ${in_file} | sed -E 's/tmp\/(.*)\.gif/\1\.mp4/')
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
for in_file in tmp/*.gif
do
    out_file=$(echo ${in_file} | sed -E 's/tmp\/(.*)\.gif/\1_opt.gif/')
    gifsicle -O3 --colors 16 --lossy=200 -o "${out_file}" "${in_file}"
done


# move gifs in tmp in originals and delete tmp:
mv -i tmp/* originals
rmdir tmp


# move *.mp4 in "${videos_dst}":
# move *_opt.gif in "${opt_gifs_dst}":
answer='NONE'
while [[ $answer != 'y' && $answer != 'n' && $answer != '' ]]; do
    read -r -p "Do you want to move newly created videos and optimized gifs in two new separate folders? ({y}|n)" answer
done
if [[ $answer == 'y' || $answer == '' ]]; then
    # create folders:
    #if [[ -e "${videos_dst}" ]]; then
    #    echo "\"${videos_dst}\" already exists"
    #    echo "Aborting."
    #    exit 1
    #fi
    #if [[ -e "${opt_gifs_dst}" ]]; then
    #    echo "\"${opt_gifs_dst}\" already exists"
    #    echo "Aborting."
    #    exit 1
    #fi
    if [[ -d "${videos_dst}" ]]; then
        echo "Warning: \"${videos_dst}\" already exists, proceeding anyway (adding files to the existing folder)..."
    fi
    if [[ -d "${opt_gifs_dst}" ]]; then
        echo "Warning: \"${opt_gifs_dst}\" already exists, proceeding anyway (adding files to the existing folder)..."
    fi
    mkdir -p "${videos_dst}"
    mkdir -p "${opt_gifs_dst}"
    # move files:
    mv -i *.mp4 "${videos_dst}"
    mv -i *_opt.gif "${opt_gifs_dst}"
    echo "Assert (you should check manually this): \"${videos_dst}\" and \"${opt_gifs_dst}\" should have the same number of elements (the number of gifs converted)"
fi



# get .gif duration and info:
# $ gifsicle --info myfile.gif  # for each frame: duration each frame, color info, image size
# $ exiftool myfile.gif  # total duration, number of frames, image size, ...
# check video FPS:
# $ mediainfo myfile.mp4 | grep -i fps






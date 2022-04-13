#!/usr/bin/env bash -x
set -x  # and call as bash clean.sh; or without set -x and call as bash -x clean.sh

#rm *.csv *.gv *.svg
#rm neat-checkpoint-*

rm -Ir frames_*
rm -I *.gif
rm -I log*.log
# ATT: se prima ho messo yes, nel secondo caso non mi chiede nulla (if they are few)!! bugg

rm -I *neat-checkpoint-*

# ask them separately, so the user can decide which delete and which not
rm -I *.gv
rm -I *.svg
rm -I *.png
rm -I *.pickle
rm -I *_config


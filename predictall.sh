#!/bin/bash
filelist=`ls data/gumed/*.mhd | sed -e 's/data\/gumed\///g' -e 's/\.mhd//g'`
for filename in $filelist; do
    echo "Running for $filename"
    python predict_raw.py $filename
done

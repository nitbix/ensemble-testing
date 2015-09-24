#!/bin/bash

outfile=`echo $1 | sed 's/\.yaml/.out/' | sed 's/^models/outputs/'`

stdbuf -i0 -e0 -o0 ./ensemble.py $1 | tee $outfile

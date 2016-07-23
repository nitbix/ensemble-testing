#!/bin/bash

first_file=$1
second_file=$2
out_file=$3
error_name=$4
title=$5
first_label=$6
second_label=$7
epochs=$8

dumped_dir=/var/www/html/dumped/graphs

echo "
#set style line 1 lc rgb '#8b1a0e' lw 2 # --- red
#set style line 2 lc rgb '#5e9c36' lw 2 # --- green
set style line 1 lc rgb '#339933' lw 2 # --- green
set style line 2 lc rgb '#0033AA' lw 2 # --- blue
set style line 11 lc rgb '#808080' lt 1
set border 3 back ls 11
set tics nomirror
set style line 12 lc rgb '#808080' lt 0 lw 1
set grid back ls 12
set title \"$title\"
set terminal postscript eps enhanced color size 4,2.5
set xlabel \"Training Epoch\"
set ylabel \"${error_name} Error (%)\"
set terminal postscript eps enhanced color size 4,2.5
set out \"${out_file}.eps\"
plot \"${first_file}\"  every ::0::${epochs} using 1:2 linestyle 1 title \"${first_label}\" with lines,\
     \"${second_file}\" every ::0::${epochs} using 1:2 linestyle 2 title \"${second_label}\" with lines,\
     \"${first_file}\" every ::0::${epochs} notitle linestyle 1 with errorbars,\
     \"${second_file}\" every ::0::${epochs} notitle linestyle 2 with errorbars

set terminal png
set out \"${out_file}.png\"
replot
" | gnuplot
mkdir -p ${dumped_dir}
cp ${out_file}.png ${dumped_dir}

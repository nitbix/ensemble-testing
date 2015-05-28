#!/bin/bash

first_file=$1
second_file=$2
out_file=$3

echo "
#set style line 1 lc rgb '#8b1a0e' lw 2 # --- red
#set style line 2 lc rgb '#5e9c36' lw 2 # --- green
set style line 1 lc rgb '#993300' lw 2 # --- orange
set style line 2 lc rgb '#003366' lw 2 # --- blue
set style line 11 lc rgb '#808080' lt 1
set border 3 back ls 11
set tics nomirror
set style line 12 lc rgb '#808080' lt 0 lw 1
set grid back ls 12
#set title \"Misclassification Rates ($technique - $problem_name)\"
set terminal postscript eps enhanced color size 4,2.5
set xlabel \"Training Epoch\"
set ylabel \"Error (%)\"
set terminal postscript eps enhanced color size 4,2.5
set out \"${out_file}.eps\"
plot \"${first_file}\"  using 1:2 lw 2 title \"\" with lines,\
     \"${second_file}\" using 1:2 lw 2 title \"\" with lines
set terminal png
set out \"${out_file}.png\"
replot
" | gnuplot

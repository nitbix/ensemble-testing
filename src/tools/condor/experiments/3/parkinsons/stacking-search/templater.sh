#!/bin/bash

cat template.csub > send-all.csub

for te in 0.01; do
for se in 0.05 0.01; do
for ste in 0.01 0.0001 0.000001; do
for rate1 in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
for rate2 in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
for mid in 10 30 100; do
	cat template.prop | sed "s/{te}/$te/" | sed "s/{se}/$se/" | sed "s/{ste}/$ste/" | sed "s/{mid}/$mid/" | sed "s/{rate1}/$rate1/" | sed "s/{rate2}/$rate2/" > $te-$se-$ste-$mid-$rate1-$rate2.prop
	echo "arguments = 3 parkinsons stacking-search/$te-$se-$ste-$mid-$rate1-$rate2" >> send-all.csub
	echo "queue 30" >> send-all.csub
done
done
done
done
done
done

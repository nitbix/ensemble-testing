#!/bin/bash

cat template.csub > send-all.csub

for te in 0.03 0.01 0.001; do
for se in 0.05 0.03 0.01; do
for rate in 0.1 0.3 0.5 0.7 0.9; do
	cat template.prop | sed "s/{te}/$te/" | sed "s/{se}/$se/" | sed "s/{rate}/$rate/" > $te-$se-$ste-$mid.prop
	echo "arguments = 3 parkinsons dropout-search/$te-$se-$ste-$mid" >> send-all.csub
	echo "queue 30" >> send-all.csub
done
done
done

#!/bin/bash

cat template.csub > send-all.csub

for te in 0.01 0.003; do
for se in 0.05 0.03; do
for ste in 0.01 0.0001 0.000001; do
for mid in 10 30 100; do
	cat template.prop | sed "s/{te}/$te/" | sed "s/{se}/$se/" | sed "s/{ste}/$ste/" | sed "s/{mid}/$mid/" > $te-$se-$ste-$mid.prop
	echo "arguments = 3 parkinsons stacking-search/$te-$se-$ste-$mid" >> send-all.csub
	echo "queue 30" >> send-all.csub
done
done
done
done

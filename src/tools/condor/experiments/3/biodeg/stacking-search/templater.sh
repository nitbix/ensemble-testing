#!/bin/bash

cat template.csub > send-all.csub

for te in 0.08 0.05; do
for se in 0.15 0.1; do
for ste in 0.1 0.01 0.001; do
for mid in 10 30 100; do
	cat template.prop | sed "s/{te}/$te/" | sed "s/{se}/$se/" | sed "s/{ste}/$ste/" | sed "s/{mid}/$mid/" | sed "s/{extra}//" > $te-$se-$ste-$mid.prop
	cat template.prop | sed "s/{te}/$te/" | sed "s/{se}/$se/" | sed "s/{ste}/$ste/" | sed "s/{mid}/$mid/" | sed "s/{extra}/-adaptive/" > $te-$se-$ste-$mid-adaptive.prop
#	echo "arguments = 3 biodeg stacking-search/$te-$se-$ste-$mid" >> send-all.csub
#	echo "queue 30" >> send-all.csub
	echo "arguments = 3 biodeg stacking-search/$te-$se-$ste-$mid-adaptive" >> send-all.csub
	echo "queue 30" >> send-all.csub
done
done
done
done

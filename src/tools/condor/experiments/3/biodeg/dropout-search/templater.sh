#!/bin/bash

cat template.csub > send-all.csub

for te in 0.1 0.03 0.01; do
for se in 0.15 0.1 0.03; do
for rate in 0.1 0.3 0.5 0.7 0.9; do
	cat template.prop | sed "s/{te}/$te/" | sed "s/{se}/$se/" | sed "s/{rate}/$rate/" > $te-$se-$rate.prop
	echo "arguments = 3 biodeg dropout-search/$te-$se-$rate" >> send-all.csub
	echo "queue 30" >> send-all.csub
done
done
done

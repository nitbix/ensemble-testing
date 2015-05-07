#!/bin/bash

cat template.csub > send-all.csub

for te in 0.01 0.03; do
for se in 0.15 0.1; do
	cat template.prop | sed "s/{te}/$te/" | sed "s/{se}/$se/" | sed "s/{rate}/$rate/" > $te-$se.prop
	echo "arguments = 3 biodeg adaboost-search/$te-$se" >> send-all.csub
	echo "queue 100" >> send-all.csub
done
done

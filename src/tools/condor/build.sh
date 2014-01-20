#!/bin/bash
cd ~/git/ensemble-testing
git pull -u
cd ~/git/encog-java-core/
git pull -u
cd ~/ensemble-testing
cp ~/git/ensemble-testing/src/tools/condor/*.bat .
cp -r ~/git/ensemble-testing/src/tools/condor/experiments .
#forced rebuild of all ensemble code
cd ~/git/ensemble-testing/src
ant build
ant dist
cp ensemble-testing.jar ~/ensemble-testing

#!/bin/bash
cd ~/git/ensemble-testing
git pull -u
cd ~/git/encog-java-core/src/main/java
git pull -u
cd ~/ensemble-testing
cp ~/git/ensemble-testing/src/tools/condor/*.csub .
cp ~/git/ensemble-testing/src/tools/condor/*.bat .
cp ~/git/ensemble-testing/src/tools/condor/experiments .
#forced rebuild of all ensemble code
for target in `find ~/git/encog-java-core/src/main/java/org/encog/ensemble -type d`; do
	javac -d ~/ensemble-testing -implicit:class -sourcepath ~/git/ensemble-testing/src:~/git/encog-java-core/src/main/java $target/*.java
done
javac -d ~/ensemble-testing -implicit:class -sourcepath ~/git/ensemble-testing/src:~/git/encog-java-core/src/main/java ~/git/encog-java-core/src/main/java/org/encog/ml/data/basic/*.java
javac -d ~/ensemble-testing -implicit:class -sourcepath ~/git/ensemble-testing/src:~/git/encog-java-core/src/main/java ~/git/ensemble-testing/src/helpers/*.java
javac -d ~/ensemble-testing -implicit:class -sourcepath ~/git/ensemble-testing/src:~/git/encog-java-core/src/main/java ~/git/ensemble-testing/src/main/*.java

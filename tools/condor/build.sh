#!/bin/bash
cd ~/git/ensemble-testing
git pull -u
cd ~/git/encog-java-core
git pull -u
cd ~/ensemble-testing
javac -d ~/ensemble-testing -implicit:class -sourcepath ~/git/ensemble-testing/src:~/git/encog-java-core/src/main/java ~/git/encog-java-core/src/main/java/org/encog/ml/data/basic/*.java
javac -d ~/ensemble-testing -implicit:class -sourcepath ~/git/ensemble-testing/src:~/git/encog-java-core/src/main/java ~/git/ensemble-testing/src/helpers/*.java
javac -d ~/ensemble-testing -implicit:class -sourcepath ~/git/ensemble-testing/src:~/git/encog-java-core/src/main/java ~/git/ensemble-testing/src/main/*.java

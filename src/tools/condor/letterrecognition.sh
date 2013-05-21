#!/bin/bash
java -cp .:mysql-connector-java-5.1.25-bin.jar main/Test $1 problems/uci_letterrecognition 1,3,10,30,100 16000 0.015 10 0.1 rprop mlp:300:sigmoid $2 false 0.03

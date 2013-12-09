net use M: \\studentnfs\amosca02
M:
cd M:\build\ensemble-testing
java -cp M:\build\ensemble-testing;M:\build\ensemble-testing\mysql-connector-java-5.1.25-bin.jar main/Test bagging problems\uci_ionosphere 1 200 0.3 10 0.1 rprop mlp:30:sigmoid majorityvoting false 0.3
C:
net use M: /delete

net use h: \\studentnfs\amosca02
cd h:\build\ensemble-testing
java -cp . main/Test %1 problems\uci_ionosphere 1 200 0.3 10 0.1 rprop%4 mlp:30:sigmoid majorityvoting false 0.3
rem net use h: /delete

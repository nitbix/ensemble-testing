#!/bin/bash

tar cvzf ensemble-testing-compiled.tar.gz main helpers data problems techniques tools
scp ensemble-testing-compiled.tar.gz amosca02@ubuntu2:

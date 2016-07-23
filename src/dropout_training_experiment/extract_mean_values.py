#!/usr/bin/python

import re
import yaml
import numpy
import os

from stats_common import *

results_table="input_update"
results_db="amosca02"                                                                                                                                                                            
results_host="gpuvm1"  

means, stdevs, methods, datasets, hist_means, hist_stdevs = extract_results(results_host, results_db, results_table)

def write_hist_file(means, stdevs, name):
    assert len(means) == len(stdevs)
    with open(name,'w') as f:
        for i in len(means):
            f.write(str(i) + " " + str(means[i] * 100) + " " + str(stdevs[i]) + "\n")

print " & " + " & ".join(methods)
for dataset in datasets:
    for method in methods:
        print "\n{0}-{1}".format(dataset,method)
        if dataset in hist_means and method in hist_means[dataset]:
            for t in hist_means[dataset][method]:
                d = "history_data/{0}/{1}/{2}".format(results_table,
                        dataset,method)
                if not os.path.exists(d):
                    os.makedirs(d)
                filename = "{0}/{1}.dat".format(d,t)
                write_hist_file(
                        hist_means[dataset][method][t],
                        hist_stdevs[dataset][method][t],
                        filename
                )
                print filename
        else:
            print "missing"

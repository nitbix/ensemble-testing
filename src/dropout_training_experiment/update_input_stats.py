#!/usr/bin/python

import re
import yaml
import math

from stats_common import *

results_table="input_update"
results_db="amosca02"                                                                                                                                                                            
results_host="gpuvm1"  

means, stdevs, methods, datasets, hist_means, hist_stdevs = extract_results(results_host, results_db, results_table)
hsep = " & "
vsep = " \\\\"
def make_line(first,mean,stdev,min_bold = False):
    it = []
    for i,x in enumerate(mean):
        if isinstance(x,float):
            it.append("{0:.2f}".format(x))
        else:
            it.append(str(x))
    mean = it
    if min_bold:
        str_mean = []
        for x in mean:
            if x == min(mean):
                str_mean.append("$\\mathbf{{ {0} }} $".format(x))
            else:
                    str_mean.append("$ {0} $".format(x))
    else:
        if len(stdev) > 0:
            str_mean = [
                    "$ {0}\% $ $ ({1:.2f})$".format(x,stdev[i])
                    for i,x in enumerate(mean)
            ]
        else:
            str_mean = ["$ {0} $".format(x) for x in mean]
    if first is not None:
        return first + hsep + hsep.join(str_mean) + vsep
    else:
        return hsep.join(str_mean) + vsep

def arrify(x):
    return [x['test'],x['valid'],x['epoch']]
#MIDDLE TABLES
for dataset in sorted(datasets):
    print "\n"
    print dataset
    print """
\\begin{table}[h]
\\centering
\\begin{tabular}
    """
    print make_line("",
            ["Mean Test Err (std)", "Mean Best Valid Err (std)", "Epochs"],
            [])
    print "\\hline"
    line = []
    for method in sorted(methods):
        if dataset in means and method in means[dataset]:
            mean = means[dataset][method]
            stdev = stdevs[dataset][method]
            print make_line(method,arrify(mean),arrify(stdev))
        else:
            print "missing \\"
    print """
\\hline
\\end{tabular}
\\end{table}
    """

#FINAL TABLE
print """
\\begin{table}[h]
\\centering
\\begin{tabular}
"""
print make_line("",sorted(methods),[])
print "\\hline"
for dataset in sorted(datasets):
    line = []
    for method in sorted(methods):
        if dataset in means and method in means[dataset]:
            line.append(means[dataset][method]['test'])
        else:
            line.append("missing")
    print make_line(dataset,line,[],True)

print """
\\hline
\\end{tabular}
\\end{table}
"""

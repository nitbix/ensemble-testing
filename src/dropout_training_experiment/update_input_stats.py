#!/usr/bin/python

import re
import yaml

results_table="input_update"
results_db="amosca02"                                                                                                                                                                            
results_host="gpuvm1"  

def clean_dataset(d):
    d = re.sub(r"/$",'',d)
    no_path = re.sub(r".*/",'',d)
    no_extension = re.sub(r".pkl.gz$",'',no_path)
    return no_extension


def clean_transform(t):
    if t is None:
        return ""
    else:
        return "-trans"

def clean_update_rule(r):
    rule_name = re.sub(r"\s*{.*",'',r).lower()
    params_str = re.sub(r".*{\s*",'',r)
    params_str = re.sub(r"\s*}.*",'',params_str)
    params_str = re.sub(r"\s*,\s*","\n",params_str)
    try:
        params = yaml.load(params_str)
    except:
        print "{0} has broken params: {1}".format(rule_name,params_str)
        exit(1)
    if 'momentum' in params and params['momentum'] != 0:
        return rule_name + "-mom"
    else:
        return rule_name

def clean_update_input(b):
    print b
    if b:
        return 'update'
    else:
        return 'noupdate'

from pymongo import MongoClient
conn = MongoClient(host=results_host)
db = conn[results_db]
table = db[results_table]


pipeline = [
    {   "$group": 
        { "_id":
            {   "params_update_rule" : "$params.update_rule",
                "params_n_hidden":"$params.n_hidden",
                "params_dataset" : "$params.dataset",
                "params_online_transform": "$params.online_transform",
                "params_learning_rate": "$params.learning_rate",
                "params_update_input": "$params.update_input"
            },
            "count": {"$sum": 1},
            "avg_best_epoch": {"$avg": "$best_epoch"},
            "avg_best_valid": {"$avg": "$best_valid"},
            "avg_best_test": {"$avg": "$best_test"}
        },
    },
    {
        "$sort":
        {
            #"params_dataset": 1,
            #"params_n_hidden": 1,
            "params.update_rule": 1,
        }
    }
]

cursor = table.aggregate(pipeline=pipeline)
means  = {}
stdevs = {}
methods = []
datasets = []

for r in cursor['result']:
    print r
    x = r['_id']
    if r['count'] < 10000:
        print "dataset: {0}".format(x['params_dataset'])
        print "arch: {0}".format(x['params_n_hidden'][0][0])
        print "transform: {0}".format(x['params_online_transform'])
        print "update_input: {0}".format(x['params_update_input'])
        print "  count: {0}".format(r['count'])
        print "  avg_best_epoch: {0}".format(r['avg_best_epoch'])
        print "  avg_best_valid: {0}".format(r['avg_best_valid'])
        print "  avg_best_test: {0}".format(r['avg_best_test'])
        #print r
        print "-----------"
    dataset = "{0}{1}-{2}".format(
            clean_dataset(x['params_dataset']),
            clean_transform(x['params_online_transform']),
            x['params_n_hidden'][0][0])
    method = "{0}".format(
            clean_update_input(x['params_update_input'])
            )
    if dataset not in datasets:
        datasets.append(dataset)
    if method not in methods:
        methods.append(method)
    if dataset not in means:
        means[dataset] = {}
    if method not in means[dataset]:
        means[dataset][method] = {}
    means[dataset][method]['test'] = r['avg_best_test']
    means[dataset][method]['valid'] = r['avg_best_valid']
    means[dataset][method]['epoch'] = r['avg_best_epoch']

hsep = " & "
vsep = " \\\\"
def make_line(first,items,min_bold = False):
    it = []
    for x in items:
        if isinstance(x,float):
            it.append("{0:.2f}".format(x))
        else:
            it.append(str(x))
    items = it
    if min_bold:
        str_items = []
        for x in items:
            if x == min(items):
                str_items.append("$\\mathbf{{ {0} }} $".format(x))
            else:
                str_items.append("$ {0} $".format(x))
    else:
        str_items = ["$ {0} $".format(x) for x in items]
    if first is not None:
        return first + hsep + hsep.join(str_items) + vsep
    else:
        return hsep.join(str_items) + vsep

#MIDDLE TABLES
for dataset in sorted(datasets):
    print "\n"
    print dataset
    print """
\\begin{table}[h]
\\centering
\\begin{tabular}
    """
    print make_line("",["Mean Test Err (%)", "Mean Best Valid Err (%)", "Epochs"])
    print "\\hline"
    line = []
    for method in sorted(methods):
        if dataset in means and method in means[dataset]:
            x = means[dataset][method]
            print make_line(method,[x['test'],x['valid'],x['epoch']])
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
print make_line("",sorted(methods))
print "\\hline"
for dataset in sorted(datasets):
    line = []
    for method in sorted(methods):
        if dataset in means and method in means[dataset]:
            line.append(means[dataset][method]['test'])
        else:
            line.append("missing")
    print make_line(dataset,line,True)

print """
\\hline
\\end{tabular}
\\end{table}
"""

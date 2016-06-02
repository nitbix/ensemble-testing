#!/usr/bin/python

import re
import yaml
import numpy
import os

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

def clean_update_rule(r,update_input=False):
    rule_name = re.sub(r"\s*{.*",'',r).lower()
    params_str = re.sub(r".*{\s*",'',r)
    params_str = re.sub(r"\s*}.*",'',params_str)
    params_str = re.sub(r"\s*,\s*","\n",params_str)
    params = yaml.load(params_str)
    append = ''
    if update_input:
        append='-update_input'
    if 'momentum' in params and params['momentum'] != 0:
        append = append + "-momentum"
    return rule_name + append

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
            "avg_best_test": {"$avg": "$best_test"},
            "train_history": {"$push": "$train_history"},
            "valid_history": {"$push": "$validation_history"},
            "test_history": {"$push": "$test_history"}
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
hist_means = {}
methods = []
datasets = []

def make_mean_history(matrix,mode):
    m = [x for x in matrix if len(x) == len(matrix[0])]
    hist_means[dataset][method][mode] = numpy.mean(numpy.asarray(m),axis=0)

for r in cursor['result']:
    x = r['_id']
    if r['count'] < 100:
        print "dataset: {0}".format(x['params_dataset'])
        print "arch: {0}".format(x['params_n_hidden'][0][0])
        print "update_rule: {0}".format(x['params_update_rule'])
        print "transform: {0}".format(x['params_online_transform'])
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
    if 'params_update_input' in x:
        method = "{0}".format(
                clean_update_rule(x['params_update_rule'],x['params_update_input']),
                )
    else:
        method = "{0}".format( clean_update_rule(x['params_update_rule']))
    if dataset not in datasets:
        datasets.append(dataset)
    if method not in methods:
        methods.append(method)
    if dataset not in means:
        means[dataset] = {}
    if method not in means[dataset]:
        means[dataset][method] = {}
    if dataset not in hist_means:
        hist_means[dataset] = {}
    if method not in hist_means[dataset]:
        hist_means[dataset][method] = {}
    means[dataset][method]['test'] = r['avg_best_test']
    if 'train_history' in r:
        make_mean_history(r['train_history'],'train')
    if 'valid_history' in r:
        make_mean_history(r['valid_history'],'valid')
    if 'test_history' in r:
        make_mean_history(r['test_history'],'test')

def write_hist_file(data,name):
    with open(name,'w') as f:
        i = 0
        for x in data:
            f.write(str(i) + " " + str(x * 100) + "\n")
            i+=1

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
                write_hist_file(hist_means[dataset][method][t],filename)
                print filename
        else:
            print "missing"

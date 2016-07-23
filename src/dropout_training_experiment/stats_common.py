import re
import yaml
import math
import numpy

from pymongo import MongoClient

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
            "sum_best_epoch": {"$sum": "$best_epoch"},
            "sum_best_valid": {"$sum": "$best_valid"},
            "sum_best_test": {"$sum": "$best_test"},
            "sq_best_epoch": {"$sum": {"$multiply":["$best_epoch", "$best_epoch"]}},
            "sq_best_valid": {"$sum": {"$multiply":["$best_valid", "$best_valid"]}},
            "sq_best_test": {"$sum": {"$multiply":["$best_test", "$best_test"]}},
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

def make_mean_history(matrix,mode):
    m = [x for x in matrix if len(x) == len(matrix[0])]
    return numpy.mean(numpy.asarray(m),axis=0)

def make_stdev_history(matrix,mode):
    m = [x for x in matrix if len(x) == len(matrix[0])]
    return numpy.std(numpy.asarray(m),axis=0)

def extract_results(results_host,results_db,results_table):
    conn = MongoClient(host=results_host)
    db = conn[results_db]
    table = db[results_table]

    cursor = table.aggregate(pipeline=pipeline)
    means  = {}
    stdevs = {}
    hist_means = {}
    hist_stdevs = {}
    methods = []
    datasets = []

    for r in cursor['result']:
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
            std_best_epoch = _std(
                    r['count'],
                    r['sum_best_epoch'],
                    r['sq_best_epoch']
            )
            std_best_valid = _std(
                    r['count'],
                    r['sum_best_valid'],
                    r['sq_best_valid']
            )
            std_best_test = _std(
                    r['count'],
                    r['sum_best_test'],
                    r['sq_best_test']
            )
            print "  std_best_epoch: {0}".format(std_best_epoch)
            print "  std_best_valid: {0}".format(std_best_valid)
            print "  std_best_test: {0}".format(std_best_test)
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
            stdevs[dataset] = {}
            hist_means[dataset] = {}
            hist_stdevs[dataset] = {}
        if method not in means[dataset]:
            means[dataset][method] = {}
            stdevs[dataset][method] = {}
            hist_means[dataset][method] = {}
            hist_stdevs[dataset][method] = {}
        means[dataset][method]['test'] = r['avg_best_test']
        means[dataset][method]['valid'] = r['avg_best_valid']
        means[dataset][method]['epoch'] = r['avg_best_epoch']
        stdevs[dataset][method]['test'] =  std_best_test
        stdevs[dataset][method]['valid'] = std_best_valid
        stdevs[dataset][method]['epoch'] = std_best_epoch
        for d in ['train', 'valid', 'test']:
            i = '{0}_history'.format(d)
            if i in r:
                hist_means[dataset][method][d] = make_mean_history(r[i],d)
                hist_stdevs[dataset][method][d] = make_stdev_history(r[i],d)
    return means, stdevs, methods, datasets, hist_means, hist_stdevs


def _std(s0,s1,s2):
    return math.sqrt(s0 * s2 - pow(s1,2)) / s0

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
    if b:
        return 'update'
    else:
        return 'noupdate'


#!/usr/bin/python

results_table="uprop_paper"
results_db="amosca02"                                                                                                                                                                            
results_host="gpuvm1"  

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
                "params_learning_rate": "$params.learning_rate"
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

cursor = db.uprop_paper.aggregate(pipeline=pipeline)

for r in cursor['result']:
    print "dataset: {0}".format(r['_id']['params_dataset'])
    print "arch: {0}".format(r['_id']['params_n_hidden'][0][0])
    print "update_rule: {0}".format(r['_id']['params_update_rule'])
    print "transform: {0}".format(r['_id']['params_online_transform'])
    print "  count: {0}".format(r['count'])
    print "  avg_best_epoch: {0}".format(r['avg_best_epoch'])
    print "  avg_best_valid: {0}".format(r['avg_best_valid'])
    print "  avg_best_test: {0}".format(r['avg_best_test'])
    #print r
    print "-----------"

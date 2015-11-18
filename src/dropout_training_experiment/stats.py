#!/usr/bin/python

import numpy
import pymongo

conn = pymongo.Connection()
db = conn.amosca02
coll = db.reverse_pretraining

errors = {}
for r in coll.find():
    if r['best_valid'] < 0.1:
        d = r['params']['dataset']
        if d not in errors:
            errors[d] = {}
        p = r['params']['pretraining']
        if p not in errors[d]:
            errors[d][p] = {'best_epoch': [], 'best_test': [], 'best_valid':[],
                    'best_test_history': [], 'best_validation_history':[]}
        for m in ['test_history', 'validation_history']:
            errors[d][p]['best_{0}'.format(m)].append(min(r[m]) * 100.)
        for m in ['best_epoch', 'best_test', 'best_valid']:
            errors[d][p][m].append(r[m])
        errors[d][p]

for d in errors.keys():
    print '{0}:'.format(d)
    for p in errors[d].keys():
        print "  {0}:".format(p)
        print "    count: {0}".format(len(errors[d][p]['best_test']))
        for m in errors[d][p].keys():
            print "    {0}:".format(m)
            data = numpy.asarray(errors[d][p][m])
            print "      mean: {0}".format(data.mean())
            print "      std : {0}".format(numpy.std(data))

#!/usr/bin/env python
# spark-submit Ch01_Lab.py
from initspark import *
sc, spark, conf, HOME = initspark()
cc = sc.textFile(HOME+'/datasets/finance/CreditCard.csv')
first = cc.first()
cc = cc.filter(lambda x : x != first)
cc.take(10)
cc = cc.map(lambda x : x.split(',')) 
cc.take(10)
cc = cc.map(lambda x : (x[0][1:], x[1][1:-1], x[5], float(x[6])))
#print (cc.collect())
print(cc.take(10))

# Bonus
cc = cc.map(lambda x : ((x[0], x[1]), (x[2], x[3])))
ccf = cc.filter(lambda x : x[1][0] == 'F').map(lambda x : (x[0], x[1][1]))
ccg = ccf.reduceByKey(lambda x, y : x + y)
#print (ccg.sortByKey().collect())
print(ccg.sortByKey().take(10))



#!/usr/bin/env python
import sys, os
PROGRAM = os.getenv("PROGRAM")
PATH = '/home/student/ROI/' + PROGRAM
sys.path.append(PATH)
from initspark import *
sc, spark, conf = initspark()
cc = sc.textFile(PATH+'/datasets/finance/CreditCard.csv')
first = cc.first()
cc = cc.filter(lambda x : x != first)
cc.take(10)
cc = cc.map(lambda x : x.split(',')) 
cc.take(10)
cc = cc.map(lambda x : ((x[0][1:], x[1][1:-1], x[5], float(x[6]))))
#print (cc.collect())
print(cc.take(10))

# Bonus
cc = cc.map(lambda x : ((x[0][1:], x[1][1:-1]), (x[5], float(x[6]))))
ccf = cc.filter(lambda x : x[1][0] == 'F').map(lambda x : (x[0], x[1][1]))
ccg = ccf.reduceByKey(lambda x, y : x + y)
#print (ccg.sortByKey().collect())
print(ccg.sortByKey().take(10))



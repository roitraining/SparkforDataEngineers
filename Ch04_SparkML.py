#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/home/student/ROI/SparkProgram')
from initspark import *
sc, spark, conf = initspark()

import pandas as pd
import matplotlib as mp
import numpy
from matplotlib import pyplot as plt
from IPython.display import display


# In[2]:


#filename = 'CarLoanDefaults.csv'

filename = 'bank.csv'
df = spark.read.csv(f'/home/student/ROI/Spark/datasets/finance/{filename}', header = True, inferSchema = True)
display(df.limit(10).toPandas())

# Save a pointer to the raw data
df0 = df


# In[3]:


def drop_columns(df, collist):
    return df.select([c for c in df.columns if c not in collist])

def auto_numeric_features(df, exceptlist = ()):
    numeric_features = [t[0] for t in df.dtypes if t[0] not in exceptlist and t[1] in ['int', 'double']]
    return numeric_features

def auto_categorical_features(df):
    categorical_features = [c for c in df.columns if c.endswith('_ID') or c.endswith('_FLAG')]
    return categorical_features

def describe_numeric_features(df, numeric_features):
    print(df.select(numeric_features).describe().toPandas().transpose())

    
if filename == 'bank.csv':
    drop_cols = ('day','month')
    target_col = 'deposit'
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
elif filename == 'CarLoanDefaults.csv':
    drop_cols = ('UNIQUEID','DATE_OF_BIRTH','DISBURSAL_DATE', 'AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH', 'MOBILENO_AVL_FLAG')
    target_col = 'LOAN_DEFAULT'
    categorical_features = auto_categorical_features(df)

df = drop_columns(df, drop_cols)
numeric_features = auto_numeric_features(df, exceptlist = categorical_features)
print (numeric_features)
print ('*' * 80)
describe_numeric_features(df, numeric_features)
print ('*' * 80)

# save a pointer to the fixed data
df1 = df


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
def scatter_matrix(df, numeric_features):
    numeric_data = df.select(numeric_features).toPandas()
    axs = pd.plotting.scatter_matrix(numeric_data, figsize=(8, 8));
    n = len(numeric_data.columns)
    for i in range(n):
        v = axs[i, 0]
        v.yaxis.label.set_rotation(0)
        v.yaxis.label.set_ha('right')
        v.set_yticks(())
        h = axs[n-1, i]
        h.xaxis.label.set_rotation(90)
        h.set_xticks(())
        
scatter_matrix(df, numeric_features)


# In[5]:


def fix_categorical_data(df, categorical_features, target_col):
    from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
    from pyspark.ml import Pipeline

    stages = []

    for c in categorical_features:
        stringIndexer = StringIndexer(inputCol = c, outputCol = c + 'Index')
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[c + "classVec"])
        stages += [stringIndexer, encoder]

    label_stringIdx = StringIndexer(inputCol = target_col, outputCol = 'label')
    stages += [label_stringIdx]

    assemblerInputs = [c + "classVec" for c in categorical_features] + numeric_features
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]

    cols = df.columns
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df)
    dfx = pipelineModel.transform(df)
    dfx = dfx.select(['label', 'features'] + cols)
    return dfx

df = fix_categorical_data(df, categorical_features, target_col)
# save a pointer to this stage of the dataframe
df2 = df

df.printSchema()


# In[6]:


df.groupBy('label').count().show()


# In[9]:


pd.DataFrame(df.take(5), columns = df.columns).transpose()


# In[10]:


train, test = df.randomSplit([.7,.3], seed = 1000)
print (f'Training set row count {train.count()}')
print (f'Testing set row count {test.count()}')
      


# In[11]:


from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr.fit(train)
print('LR Trained')


# In[12]:


def beta_coefficients(model):
    import matplotlib.pyplot as plt
    import numpy as np
    beta = np.sort(model.coefficients)
    plt.plot(beta)
    plt.ylabel('Beta Coefficients')
    plt.show()
    
beta_coefficients(lrModel)


# In[16]:


def roc_curve(model):
    summary = model.summary
    roc = summary.roc.toPandas()
    plt.plot(roc['FPR'],roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    print('Training set area Under ROC: {}'.format(summary.areaUnderROC))

roc_curve(lrModel)


# In[17]:


def precision_recall(model):
    summary = model.summary
    pr = summary.pr.toPandas()
    plt.plot(pr['recall'],pr['precision'])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()

precision_recall(lrModel)    


# In[18]:


def evaluate_ROC(predictions):
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    evaluator = BinaryClassificationEvaluator()
    return evaluator.evaluate(predictions)

def show_predictions(predictions, limit = 20):
    print('Test Area Under ROC {}'.format(evaluate_ROC(predictions)))
    predictions.groupBy('prediction').count().show()
    predictions.select('label', 'rawPrediction', 'prediction', 'probability').show(limit)



predictions = lrModel.transform(test)
show_predictions(predictions)


# In[19]:


def better_collect(df):
    return [tuple(row) if len(tuple(row)) > 1 else tuple(row)[0] for row in df.collect()]

print (better_collect(predictions.select('prediction').distinct()))

def cm_percent(cm, length, legend = True):
    import numpy as np
    x = np.ndarray(shape = (2,2),                       buffer = np.array([100 *(cm[0][0] + cm[1][1])/length,                       100 * cm[0][1]/length, 100 * cm[1][0]/length,                       100 * (cm[1][0] + cm[0][1])/length]))
    return x


# In[23]:


def evaluate_model(model):
    beta_coefficients(lrModel)
    roc_curve(lrModel)
    precision_recall(lrModel)    

def evaluate_predictions(predictions, show = True):
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
    log = {}

    evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')
    log['auroc'] = evaluator.evaluate(predictions)  
    
    # Show Validation Score (AUPR)
    evaluator = BinaryClassificationEvaluator(metricName='areaUnderPR')
    log['aupr'] = evaluator.evaluate(predictions)

    # Metrics
    predictionRDD = predictions.select(['label', 'prediction'])                    .rdd.map(lambda line: (line[1], line[0]))
    metrics = MulticlassMetrics(predictionRDD)
    

    # Overall statistics
    log['precision'] = metrics.precision()
    log['recall'] = metrics.recall()
    log['F1 Measure'] = metrics.fMeasure()
    
    # Statistics by class
    distinctPredictions = better_collect(predictions.select('prediction').distinct())
    for x in sorted(distinctPredictions):
        log[x] = {}
        log[x]['precision'] = metrics.precision(x)
        log[x]['recall'] = metrics.recall(x)
        log[x]['F1 Measure'] = metrics.fMeasure(x, beta = 1.0)

    # Confusion Matrix
    log['cm'] = metrics.confusionMatrix().toArray()
    log['cmpercent'] = cm_percent(log['cm'], predictions.count(), show)

    if show:
        show_predictions(predictions)

        print("Area under ROC = {}".format(log['auroc']))
        print("Area under AUPR = {}".format(log['aupr']))
        print('\nOverall\ntprecision = {}\nrecall = {}\nF1 Measure = {}\n'.format( 
              log['precision'], log['recall'], log['F1 Measure']))

        for x in sorted(distinctPredictions):
            print('Label {}\ntprecision = {}\nrecall = {}\nF1 Measure = {}\n'.format( 
                  x, log[x]['precision'], log[x]['recall'], log[x]['F1 Measure']))
        
        print ('Confusion Matrix')
        print (log['cm'])
        print (' PC', 'FP\n', 'FN', 'PW')
        print (log['cmpercent'])

    return log    


log = evaluate_predictions(predictions)
print ()
print (log)


# In[ ]:


from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dtModel = dt.fit(train)
predictions = dtModel.transform(test)
predictions.select('label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[ ]:


evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC: {}'.format(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


# In[ ]:


from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions.select('label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[ ]:


evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC: {}'.format(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


# In[ ]:


from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxIter=10)
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)
predictions.select('label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[ ]:


evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC: {}'.format(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
print(gbt.explainParams())


# ## Read in a file to a Spark DataFrame

# In[ ]:





# In[ ]:



from pyspark.ml.feature import StringIndexer
def value_counts(df, cols):
    if len(cols) == 1:
        return tuple(map(tuple, df.groupBy(cols).count().orderBy('count', ascending = False).collect()))
    else:
        return [tuple(map(tuple, df.groupBy(c).count().orderBy('count', ascending = False).collect())) for c in cols]
    #return tuple(adult1.select(col).distinct().collect())

print (value_counts(adult1, ['education']))
print (value_counts(adult1, ['sex', 'race']))



#indexer = StringIndexer(inputCol="workclass", outputCol="workclassindex")
#adult = indexer.fit(adult1).transform(adult1).drop('workclass')
catcols = ['workclass', 'education','maritalstatus','occupation','relationship','race','sex','nativecountry']
#for col in catcols:
#    print(list(adult1.select(col).distinct().collect()))

#StringIndexer(inputCol='workclass', outputCol='workclassindex').fit(adult1).transform(adult1).drop('workclass').show(2)

#for col in catcols:
#    adult2 = StringIndexer(inputCol=col, outputCol=col+"index").fit(adult2).transform(adult2).drop(col)

#adult2.show()


#from pyspark.ml.feature import OneHotEncoderEstimator

#encoder = OneHotEncoderEstimator(inputCols=["workclass", "education"],
#                                 outputCols=["workclassVec", "educationVec"])
#model = encoder.fit(adult1)
#encoded = model.transform(adult1)
#encoded.show()


# ## Use createOrReplaceTempView to create a virtual table in the Hive catalog and then it can be queried using SQL as if it were a hive table

# In[ ]:





# In[ ]:


territories.createOrReplaceTempView('territories')
t1 =spark.sql('select * from territories where regionid = 1')
t1.show()
print(t1.count())


# ## Spark DataFrames can be saved to a Hive table using either the saveAsTable method or writing a SQL query that uses CREATE TABLE AS

# In[ ]:


get_ipython().system(' hadoop fs -rm -r /user/hive/warehouse/territories2')
get_ipython().system(' hadoop fs -rm -r /user/hive/warehouse/territories3')
get_ipython().system(' hadoop fs -rm -r /user/hive/warehouse/territoryregion')

territories.write.saveAsTable('Territories2', mode='overwrite')
spark.sql('create table Territories3 as select * from territories')


# ## Queries use standard HQL to mix Hive tables and virtual tables. Both are read into a Spark DataFrame and the processing happens at the Spark level not at the Hive level. HQL is just used to parse the logic into the corresponding Spark methods

# In[ ]:


sql = """select r.regionid, r.regionname, t.territoryid, t.territoryname 
from regions as r 
join territories as t on r.regionid = t.regionid 
order by r.regionid, t.territoryid"""
rt = spark.sql(sql)
rt.show(10)

tr = regions.join(territories, regions.regionid == territories.RegionID).      select('regions.regionid', 'regionname', 'TerritoryID', 'TerritoryName')
tr.show(10)


# ## Lab: Read the northwind JSON products and make it into a TempView and do the same with the CSVHeaders version of categories

# In[ ]:





# ## Install the MySQL Python connector. This has nothing to do with Spark but if you want to run SQL queries directly it is helpful.

# In[ ]:


get_ipython().system(' pip install mysql-connector-python')


# ## Let's make sure we have a database for northwind and no regions table

# In[ ]:


import mysql.connector
try:
    cn = mysql.connector.connect(host='localhost', user='test', password='password')
    cursor = cn.cursor()
    cursor.execute('create database if not exists northwind')
    cn.close()

    cn = mysql.connector.connect(host='localhost', user='test', password='password', database='northwind')
    cursor = cn.cursor()    
    cursor.execute('drop table if exists regions')
    cn.close()
except:
    print('something went wrong')
else:
    print('success')


# ## Write a DataFrame to a SQL database

# In[ ]:


regions.write.format("jdbc").options(url="jdbc:mysql://localhost/northwind", driver='com.mysql.jdbc.Driver', dbtable='regions', user='test', password = "password", mode = "append", useSSL = "false").save()


# ## Read a SQL table into a Spark DataFrame

# In[ ]:


regions2 = spark.read.format("jdbc").options(url="jdbc:mysql://localhost/northwind", driver="com.mysql.jdbc.Driver", dbtable= "regions", user="test", password="password").load()
regions2.show()


# ## Creating the regions2 DataFrame does not execute anything yet, but by making the DataFrame into a Temp View then running a Spark SQL query, it tells Spark to read the SQL data into a DataFrame and then use the cluster to do the processing, not the SQL source

# In[ ]:


regions2.createOrReplaceTempView('regions2')
spark.sql('select * from regions2 where regionid < 3').show()


# ## Alternate ways to code a query using SQL and methods

# In[ ]:


print(spark.sql('select count(*) from regions').collect())
spark.sql('select * from regions').count()


# ## Using SQL you can use familiar syntax instead of withColumn or withCoumnRenamed methods

# In[ ]:


t1 = spark.sql('select TerritoryID as TerrID, UPPER(TerritoryName) as TerritoryName, RegionID from territories')
t1.show(5)

from pyspark.sql.functions import expr
territories.withColumn('TerritoryName', expr('UPPER(TerritoryName)')).withColumnRenamed('TerritoryID', 'TerrID').show(5)


# ## Sometimes there is a function in Python that doesn't exist in SQL and it would be helpful to use, so you could make a udf and use withColumn

# In[ ]:


from pyspark.sql.functions import expr, udf
from pyspark.sql.types import *

t2 = spark.sql('select * from territories')
t2.printSchema()
#t2.show()
t2 = t2.withColumn('upperName', expr('UPPER(TerritoryName)'))
t2.show(5)

t2 = t2.withColumn('titleName', udf(lambda x : x.title(), StringType())(t2.upperName))
t2.show(5)


# ## To make it easier though, you could make the Python function into a udf that SQL can understand similar to how you can make a DataFrame seem like a virtual table with createOrReplaceTempView

# In[ ]:


def reverseString(x):
    return x[::-1]

spark.udf.register('reverse', reverseString, StringType())

spark.sql('select *, reverse(TerritoryName) as Reversed from Territories').orderBy('Reversed').show()


# ## HQL has collect_set and collect_list functions to aggregate items into a list instead of summing them up 

# In[ ]:


from pyspark.sql.functions import collect_list
territories.groupBy(territories.RegionID).agg(collect_list(territories.TerritoryName)).show()

tr1 = spark.sql("SELECT RegionID, collect_list(TerritoryName) AS TerritoryList FROM Territories GROUP BY RegionID")
tr1.show()
tr1.printSchema()
print(tr1.take(1))


# ## Instead of a simple datatype you could also collect complex structured objects using the HQL NAMED_STRUCT

# In[ ]:



sql = """SELECT r.RegionID, r.RegionName
, COLLECT_SET(NAMED_STRUCT("TerritoryID", TerritoryID, "TerritoryName", TerritoryName)) AS TerritoryList
FROM Regions AS r
JOIN Territories AS t ON r.RegionID = t.RegionID
GROUP BY r.RegionID, r.RegionName
ORDER BY r.RegionID"""

tr2 = spark.sql(sql)
tr2.printSchema()
print(tr2)
tr2.show()
print(tr2.take(2))
tr2.write.json('TerritoryRegion.json')
spark.sql('create table TerritoryRegion as ' + sql)


# ## If you have data that is already collected into a complex datatype and want to flatten it, you could use HQL EXPLODE function

# ## You could use the Spark explode method

# In[ ]:


from pyspark.sql.functions import explode
tr1.select('RegionID', explode('TerritoryList')).show()


# ## Or if the DataFrame is turned into a Temp View you could use the HQL query to do it

# In[ ]:


tr1.createOrReplaceTempView('RegionTerritories')
sql = """SELECT RegionID, TerritoryName
FROM RegionTerritories
LATERAL VIEW EXPLODE(TerritoryList) EXPLODED_TABLE AS TerritoryName
ORDER BY RegionID, TerritoryName
"""
spark.sql(sql).show()


# ## Or you could select specific elements from a collection

# In[ ]:


tr2.createOrReplaceTempView('RegionTerritories')
spark.sql("select RegionId, RegionName, TerritoryList[0] as First, TerritoryList[size(TerritoryList) - 1] as Last, size(TerritoryList) as TerritoryCount from RegionTerritories").show()


# ## If the array is of structs note the syntax of fetching the elements from the struct uses the . like an object property

# In[ ]:


sql = """SELECT RegionID, RegionName, Territory.TerritoryID AS TerritoryID
, Territory.TerritoryName AS TerritoryName
FROM RegionTerritories
LATERAL VIEW EXPLODE(TerritoryList) EXPLODED_TABLE AS Territory
"""
spark.sql(sql).show()


# ## Homework ##
# 
# ** First Challenge **
# Create a Python function to determine if a number is odd or even and use that to select only the even numbered shippers from the TSV folder of northwind. Note the TSV file does not have headers so you will need to do something to make the DataFrame have a meaningful structure. I would suggest using SparkSql as much as possible to rename and cast the columns which are ShipperID, CompanyName and Phone
# 
# ** Second Challenge **
# Take the Order_LineItems.json folder, read it into a DataFrame and flatten it and then calculate the average price paid for a product.
# 
# 

# In[ ]:


# Read the following code and see how it will shape order line items into the order header record
# You will use the result of this saved file for the second challenge
o = spark.read.csv('/home/student/ROI/Spark/datasets/northwind/CSVHeaders/orders', header = True, inferSchema = True)
od = spark.read.csv('/home/student/ROI/Spark/datasets/northwind/CSVHeaders/orderdetails', header = True, inferSchema = True)

o.createOrReplaceTempView('Orders')
od.createOrReplaceTempView('OrderDetails')
sql = """
select o.OrderID, o.CustomerID, o.OrderDate
           , COLLECT_SET(NAMED_STRUCT("ProductID", od.ProductID, 
                                      "UnitPrice", od.UnitPrice,
                                      "Quantity", od.Quantity,
                                      "Discount", od.discount)) as LineItems
from Orders as o join OrderDetails as od on o.OrderID = od.OrderID
GROUP BY o.OrderID, o.CustomerID, o.OrderDate
ORDER BY o.OrderID"""
od2 = spark.sql(sql)
od2.write.json('Orders_LineItems.json')


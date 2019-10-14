# you should make sure you have spark in your python path as below
# export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/build:$PYTHONPATH
# but if you don't it will append it automatically for this session

import platform, os, sys
from os.path import dirname

sys.path.append('/home/student/ROI/Spark')

if not 'SPARK_HOME' in os.environ and not os.environ['SPARK_HOME'] in sys.path:
    sys.path.append(os.environ['SPARK_HOME']+'/python')

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *

def initspark(appname = "Test", servername = "local", cassandra="127.0.0.1", mongo="mongodb://127.0.0.1/classroom"):
    print ('initializing pyspark')
    conf = SparkConf().set("spark.cassandra.connection.host", cassandra).setAppName(appname).setMaster(servername)
    sc = SparkContext(conf=conf)
    spark = SparkSession.builder.appName(appname) \
    .config("spark.mongodb.input.uri", mongo) \
    .config("spark.mongodb.output.uri", mongo) \
    .enableHiveSupport().getOrCreate()
    sc.setLogLevel("ERROR")
    print ('pyspark initialized')
    return sc, spark, conf

if __name__ == '__main__':
    sc, spark, conf = initspark()

def drop_columns(df, collist):
    return df.select([c for c in df.columns if c not in collist])

def auto_numeric_features(df, exceptlist = ()):
    numeric_features = [t[0] for t in df.dtypes if t[0] not in exceptlist and t[1] in ['int', 'double']]
    return numeric_features

def auto_categorical_features(df):
    categorical_features = [c for c in df.columns if c.endswith('_ID') or c.endswith('_FLAG')]
    return categorical_features

def describe_numeric_features(df, numeric_features):
    print(df.select(numeric_features).describe().toPandas())

def scatter_matrix(df, numeric_features):
    import pandas as pd
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

def fix_categorical_data(df, categorical_features, target_col, numeric_features = []):
    from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler, StringIndexerModel
    from pyspark.ml import Pipeline

    stages = []

    for c in categorical_features:
        stringIndexer = StringIndexer(inputCol = c, outputCol = c + '_Index')
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[c + "_classVec"])
        stages += [stringIndexer, encoder]

    label_stringIdx = StringIndexer(inputCol = target_col, outputCol = 'label')
    stages += [label_stringIdx]

    assemblerInputs = [c + "_classVec" for c in categorical_features] + numeric_features
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]

    cols = df.columns
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df)
    dfx = pipelineModel.transform(df)
    dfx = dfx.select(['label', 'features'] + cols)
    catindexes = {x.getOutputCol() : x.labels for x in pipelineModel.stages if isinstance(x, StringIndexerModel)}
    return dfx, catindexes

def beta_coefficients(model):
    import matplotlib.pyplot as plt
    import numpy as np
    beta = np.sort(model.coefficients)
    plt.plot(beta)
    plt.ylabel('Beta Coefficients')
    plt.show()

def roc_curve(model):
    from matplotlib import pyplot as plt
    summary = model.summary
    roc = summary.roc.toPandas()
    plt.plot(roc['FPR'],roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    print('Training set area Under ROC: {}'.format(summary.areaUnderROC))

def precision_recall(model):
    from matplotlib import pyplot as plt
    summary = model.summary
    pr = summary.pr.toPandas()
    plt.plot(pr['recall'],pr['precision'])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()

def evaluate_ROC(predictions):
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    evaluator = BinaryClassificationEvaluator()
    return evaluator.evaluate(predictions)

def show_predictions(predictions, limit = 20):
    print('Test Area Under ROC {}'.format(evaluate_ROC(predictions)))
    predictions.groupBy('prediction').count().show()
    predictions.select('label', 'rawPrediction', 'prediction', 'probability').show(limit)

def collect_tuple(df):
    return [tuple(row) if len(tuple(row)) > 1 else tuple(row)[0] for row in df.collect()]

def collect_dict(df):
    return dict(collect_tuple(df))

def cm_percent(cm, length, legend = True):
    import numpy as np
    x = np.ndarray(shape = (2,2),                       buffer = np.array([100 *(cm[0][0] + cm[1][1])/length,                       100 * cm[0][1]/length, 100 * cm[1][0]/length,                       100 * (cm[1][0] + cm[0][1])/length]))
    return x

def evaluate_model(model):
    try:
        beta_coefficients(model)
    except:
        pass
    try:    
        roc_curve(model)
    except:
        pass
    try:
        precision_recall(model)    
    except:
        pass

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
    distinctPredictions = collect_tuple(predictions.select('prediction').distinct())
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

def predict_and_evaluate(model, test, show = True):
    predictions = model.transform(test)
    if show:
        evaluate_model(model)
    log = evaluate_predictions(predictions, show)
    return (predictions, log)

def StringIndexEncode(df, columns):
    from pyspark.ml.feature import StringIndexer
    df1 = df
    for col in columns:
        indexer = StringIndexer(inputCol = col, outputCol = col+'_Index')
        df1 = indexer.fit(df1).transform(df1).drop(col) 
    return df1

def OneHotEncode(df, columns):
    from pyspark.ml.feature import OneHotEncoderEstimator
    df1 = df
    for col in columns:
        encoder = OneHotEncoderEstimator(inputCols=[col + '_Index'], outputCols=[col+'_Vector'])
        df1 = encoder.fit(df1).transform(df1).drop(col + '_Index')
    return df1

def AssembleFeatures(df, categorical_features, numeric_features, target_label = None, target_is_categorical = True):
    from pyspark.ml.feature import VectorAssembler

    assemblerInputs = [c + "_Vector" for c in categorical_features] + numeric_features
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    if target_label:
        return assembler.transform(df).withColumnRenamed(target_label, 'label' if target_is_categorical else 'target').drop(*(numeric_features + [c + '_Vector' for c in categorical_features]))
    return assembler.transform(df).drop(*(numeric_features + [c + '_Vector' for c in categorical_features]))
    
def MakeMLDataFrame(df, categorical_features, numeric_features, target_label = None, target_is_categorical = True):
    if target_is_categorical:
       df1 = StringIndexEncode(df, categorical_features + [target_label])
       df2 = OneHotEncode(df1, categorical_features)
       df3 =  AssembleFeatures(df2, categorical_features, numeric_features, target_label + '_Index')
    elif target_label:
       df1 = StringIndexEncode(df, categorical_features)
       df2 = OneHotEncode(df1, categorical_features)
       df3 =  AssembleFeatures(df2, categorical_features, numeric_features, target_label, False)
    else:
       df1 = StringIndexEncode(df, categorical_features)
       df2 = OneHotEncode(df1, categorical_features)
       df3 =  AssembleFeatures(df2, categorical_features, numeric_features)
    return df3

def display(df, limit = 10):
    from IPython.display import display    
    display(df.limit(limit).toPandas())

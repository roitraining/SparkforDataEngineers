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

def beta_coefficients(model):
    import matplotlib.pyplot as plt
    import numpy as np
    beta = np.sort(model.coefficients)
    plt.plot(beta)
    plt.ylabel('Beta Coefficients')
    plt.show()

def roc_curve(model):
    summary = model.summary
    roc = summary.roc.toPandas()
    plt.plot(roc['FPR'],roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    print('Training set area Under ROC: {}'.format(summary.areaUnderROC))

def precision_recall(model):
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

def better_collect(df):
    return [tuple(row) if len(tuple(row)) > 1 else tuple(row)[0] for row in df.collect()]

def cm_percent(cm, length, legend = True):
    import numpy as np
    x = np.ndarray(shape = (2,2),                       buffer = np.array([100 *(cm[0][0] + cm[1][1])/length,                       100 * cm[0][1]/length, 100 * cm[1][0]/length,                       100 * (cm[1][0] + cm[0][1])/length]))
    return x

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


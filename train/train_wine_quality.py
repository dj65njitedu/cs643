from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest
from pyspark import SparkConf, SparkContext
from numpy import array

conf = SparkConf().setAppName("train_wine_quality")
sc = SparkContext(conf = conf)

#Only select ['alcohol', 'free sulfur dioxide', 'pH', 'sulphates', 'volatile acidity']
def selectFeatures(row):
    return [row[1], row[5], row[8], row[9], row[10]]



rawTrainingData = sc.textFile("hdfs:///wine-quality/TrainingDataset.csv")
columnNameRow = rawTrainingData.first()
rawTrainingValues = rawTrainingData.filter(lambda x:x != columnNameRow)

#
trainingCsvData = rawTrainingValues.map(lambda x: x.split(";"))


trainingData = trainingCsvData.map(lambda row: LabeledPoint(row[-1], selectFeatures(row)))



rawValidationData = sc.textFile("hdfs:///wine-quality/ValidationDataset.csv")
columnNameRow = rawValidationData.first()
rawValidationValues = rawValidationData.filter(lambda x:x != columnNameRow)


validationCsvData = rawValidationValues.map(lambda x: x.split(";"))


validationFeatures = validationCsvData.map(lambda row: selectFeatures(row))
validationLabels = validationCsvData.map(lambda row: row[-1])
									 
model = RandomForest.trainClassifier(trainingData, numClasses=10, 
									 categoricalFeaturesInfo={}, numTrees = 100, 
									 featureSubsetStrategy='auto', impurity='gini', 
									 maxDepth=15, maxBins=100, seed=143)									 

									 
predictions = model.predict(validationFeatures)
collectedValidationLabels = validationLabels.collect()
print('Wine prediction:')
correct = 0
collectedPredictions = predictions.collect()
for i in range(len(collectedPredictions)):
	if collectedPredictions[i] == float(collectedValidationLabels[i]):
		correct = correct + 1
model.save(sc,'hdfs:///wine-quality/model')	
print(f"Accuracy: {correct/len(collectedPredictions)} Predictions Length: {len(collectedPredictions)}")
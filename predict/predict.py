import sys
import requests
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark import SparkConf, SparkContext

#Only select ['alcohol', 'free sulfur dioxide', 'pH', 'sulphates', 'volatile acidity']
def selectFeatures(row):
    return [row[1], row[5], row[8], row[9], row[10]]

url = sys.argv[1]
conf = SparkConf().setMaster("local")
sc = SparkContext(conf = conf)

r = requests.get(url, allow_redirects=True)
open('data.csv', 'wb').write(r.content)

rawValidationData = sc.textFile('data.csv')
columnNameRow = rawValidationData.first()
rawValidationValues = rawValidationData.filter(lambda x:x != columnNameRow)

validationCsvValues = rawValidationValues.map(lambda x: x.split(";"))
validationCsvValues = validationCsvValues.collect()

model_location = 'model'
model = RandomForestModel.load(sc, model_location)

predictionAndLabels = []
for row in validationCsvValues:
    predictionAndLabels.append((model.predict(selectFeatures(row)), float(row[-1])))

metrics = MulticlassMetrics(sc.parallelize(predictionAndLabels))
print("\n\n")
f1Score = metrics.fMeasure(5.0)
print(f'F-Score: {f1Score}')
same = 0
for x in predictionAndLabels:
    if x[0] == x[1]:
         same = same + 1
print(f'Accuracy: {same/len(predictionAndLabels)}')
print("\n\n")

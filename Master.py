import os
from pyspark import SparkConf, SparkContext
import csv
import math
import numpy as np
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
import matplotlib.pyplot as plt

conf = SparkConf().setMaster("local").setAppName("YourApp")

with open('/mnt/shared/Shared_Alex/meta/all_sites.csv', 'rU') as csvfile:
    rows = csv.reader(csvfile)
    meta_rows=[]
    for row in rows:
        meta_rows.append(row)
    

def distance_on_unit_sphere(lat1, long1, lat2, long2):

    degrees_to_radians = math.pi/180.0
     
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
     
    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
     
 
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + 
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )

    return arc  
def parse(dt):
    dt=dt.split(' ')
    mdy=dt[0].split('-')
    year=mdy[0]
    month = mdy[1]
    day =mdy[2]
    time_raw=dt[1].split(":")
    time= int(time_raw[0])*60 + int(time_raw[1])
    key= str(month)+":"+str(day)
    return key, time
    
def meta(id):
    cities = [[40.712682, -74.018820, 'NYNEWYOR.txt'],[41.874649, -87.639458,  'ILCHICAG.txt'],[33.747161, -84.385242, 'GAATLANT.txt'],[38.907364, -77.041689, 'TXDALLAS.txt'],[39.736359, -104.993471,  'CODENVER.txt'],[37.773713, -122.413396, 'CASANFRA.txt'],[32.718319, -117.150782,  'CASANDIE.txt'],[45.521753, -122.693904,  'ORPORTLA.txt'],[39.099380, -94.565420, 'MOKANCTY.txt']]    
    for row in meta_rows:
        if id == row[0]:
            industry=row[1]
            sq_ft=row[3]
            lat=row[4]
            long=row[5]
            distances=[]
            for city in cities:
                distances.append(distance_on_unit_sphere(float(lat),float(long),city[0],city[1]))
            file=cities[distances.index(min(distances))][2]
            output= [sq_ft,industry,file]
            return sq_ft,industry,file

def format(line):
    key, time =parse(line.split(',')[1]) 
    use=line.split(',')[2]
    return  (str(file)+":"+str(key),[use,time, sq_ft,industry])
    
def spit(x):
    print str(x)

def parse_temp(line):
    count=0
    for word in line:
        if word !='':
            count+=1
            if count ==3:
                return word
def parse_temp2(key1,line):
    count=0
    for word in line:
        if word !='':
            count+=1
            if count ==1:
                month = word
                 #REFORMAT MONTH
                if len(month)==1:
                    month = '0'+month
            if count ==2:
                day = word
                #REFORMAT DAY
                if len(day)==1:
                    day = '0'+day
            if count ==4:
                temp=word
    return (key1+":"+str(month) + ":" +str(day), [temp])
                
def clean(line):
    data = []   
    for i in line[1][0]:
        data.append(i)    
    data.append(line[1][1][0])
    return data

def extract_features_dt(record):
    cat=classify[record.pop(-2)]
    num_vect= np.array([float(field) for field in record[1:]]) 
    return np.append(cat,num_vect)

def extract_label(record):
    return float(record[0])

def evaluate_dt(train, test, maxDepth, maxBins):
    dt_model = DecisionTree.trainRegressor(train, categoricalFeaturesInfo={0:4},impurity='variance', maxDepth=maxDepth, maxBins=maxBins)
    dt_predictions = dt_model.predict(test.map(lambda x: x.features))
    dt_labelsAndPredictions = test.map(lambda lp: lp.label).zip(dt_predictions)
    dt_testMSE = dt_labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
    return dt_testMSE

sc = SparkContext(conf = conf)
ids=os.listdir('/mnt/shared/Shared_Alex/csv')
#hdfs://localhost:8020
root="/mnt/shared/Shared_Alex"
cnt=0
for i in range(len(ids)):
    if '.csv' in ids[i]:
        id = ids[i].split('.')
        sq_ft,industry,file=meta(id[0])
        if cnt==0:
            master_file = sc.textFile(root+"/csv/"+ids[i]).filter(lambda line: line.split(',')[0]!="timestamp").map(lambda line: format(line))           
        else:
            master_file= sc.textFile(root+"/csv/"+ids[i]).filter(lambda line: line.split(',')[0]!="timestamp").map(lambda line: format(line)).union(master_file)
        cnt+=1

classify = {'Commercial Property':0,'Education':1,'Food Sales & Storage':2,'Light Industrial':3}

ny=sc.textFile(root+"/allsites/NYNEWYOR.txt").filter(lambda line: parse_temp(line.split(' '))=='2012').map(lambda line: parse_temp2('NYNEWYOR.txt',line.split(' ')))
chi=sc.textFile(root+"/allsites/ILCHICAG.txt").filter(lambda line: parse_temp(line.split(' '))=='2012').map(lambda line: parse_temp2('ILCHICAG.txt',line.split(' ')))
atl=sc.textFile(root+"/allsites/GAATLANT.txt").filter(lambda line: parse_temp(line.split(' '))=='2012').map(lambda line: parse_temp2('GAATLANT.txt',line.split(' ')))
dal=sc.textFile(root+"/allsites/TXDALLAS.txt").filter(lambda line: parse_temp(line.split(' '))=='2012').map(lambda line: parse_temp2('TXDALLAS.txt',line.split(' ')))
den=sc.textFile(root+"/allsites/CODENVER.txt").filter(lambda line: parse_temp(line.split(' '))=='2012').map(lambda line: parse_temp2('CODENVER.txt',line.split(' ')))
sf=sc.textFile(root+"/allsites/CASANFRA.txt").filter(lambda line: parse_temp(line.split(' '))=='2012').map(lambda line: parse_temp2('CASANFRA.txt',line.split(' ')))
sd=sc.textFile(root+"/allsites/CASANDIE.txt").filter(lambda line: parse_temp(line.split(' '))=='2012').map(lambda line: parse_temp2('CASANDIE.txt',line.split(' ')))
por=sc.textFile(root+"/allsites/ORPORTLA.txt").filter(lambda line: parse_temp(line.split(' '))=='2012').map(lambda line: parse_temp2('ORPORTLA.txt',line.split(' ')))
ks=sc.textFile(root+"/allsites/MOKANCTY.txt").filter(lambda line: parse_temp(line.split(' '))=='2012').map(lambda line: parse_temp2('MOKANCTY.txt',line.split(' ')))

all_cities=ny.union(chi).union(atl).union(dal).union(den).union(sf).union(sd).union(por).union(ks)

master_file= master_file.leftOuterJoin(all_cities).map(lambda line: clean(line))

count=master_file.count()
#master_file.cache()
data=master_file.map(lambda r: LabeledPoint(extract_label(r),extract_features_dt(r)))



first= data.first()
(trainingData, testData) = data.randomSplit([0.8, 0.2])
dt_model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo={0:4},impurity='variance', maxDepth=7, maxBins=32)
dt_predictions = dt_model.predict(testData.map(lambda x: x.features))
dt_labelsAndPredictions = testData.map(lambda lp: lp.label).zip(dt_predictions)
dt_testMSE = dt_labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())                                    

#dt_model.save(sc, root)

print "First record :", first
print "Num Records: ", count
print "MSE: ", dt_testMSE

print dt_model.predict(np.array([3,720,90000,68]))

#print('Decision Tree Test Mean Squared Error = ' + str(dt_testMSE))

#params = [1, 2, 3, 4, 5, 10, 20]
#metrics = [evaluate_dt(trainingData, testData, param, 32) for param in params]
#print params
#print metrics
#plot(params, metrics)
#fig = plt.gcf()

#params = [2, 4, 8, 16, 32, 64, 100]
#metrics = [evaluate_dt(trainingData, testData, 5, param) for param in params]
#print params
#print metrics
#plot(params, metrics)
#fig = plt.gcf()

            



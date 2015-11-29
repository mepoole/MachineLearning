import random
from sklearn.datasets import load_iris
from numpy import linalg as LA
from sklearn.decomposition import PCA
import sys
import math
from sklearn.neighbors import LSHForest
from sklearn import preprocessing
import os
import numpy as np
from tempfile import TemporaryFile
from sklearn import svm

# center data and set within range [-1, +1]
def normalize(data):
	preprocessing.scale(data, 0, True, False, False)
	preprocessing.MinMaxScaler((-1, 1), False).fit_transform(data)
	return data


def onlineKMeans(numCentroids, batchSize, num_iter, data):
	random.seed(52686)
	centroids = random.sample(data, numCentroids)
	centerCounts = []
	for i in range(0, numCentroids):
		centerCounts.append(0.0)
	for i in range(0, num_iter):
		miniBatchCentroidAssignment =[]
		miniBatch = random.sample(data, batchSize)
		for point in miniBatch:
			bestCentroid = 0
			bestDist =sys.maxint
			for centroid in enumerate(centroids):
				dist = LA.norm(centroid[1]-point)
				if dist<bestDist:
					bestDist = dist
					bestCentroid = centroid[0]
			miniBatchCentroidAssignment.append(bestCentroid)
		for point in enumerate(miniBatch):
			centroid =miniBatchCentroidAssignment[point[0]]
			centerCounts[centroid] = centerCounts[centroid]+1
			eta = 1/centerCounts[centroid]
			centroids[centroid] = (1-eta)*centroids[centroid]+eta*point[1]
		#for centroid in centroids:
			#print(centroid)
		return np.array(centroids)

def VLAD(centroids, data):
	VLADVectors = []
	for point in data:		
		VLADVector =[0.0]*len(centroids)
		lshf = LSHForest(random_state=42)
		lshf.fit(centroids)
		#get distance using approximate nearest neighbors
		distances, centroidIndex = lshf.kneighbors(point, n_neighbors=len(centroids))
		for index, centroid in enumerate(centroidIndex[0]):
			VLADVector[centroid] = distances[0][index]
		VLADVector = np.array(VLADVector)
		VLADVectors.append(VLADVector)
	VLADVectors = np.array(VLADVectors)
		#apply pca
	pca = PCA(whiten=True)
	print(VLADVectors.shape)
	VLADVectors = pca.fit_transform(VLADVectors)
	print(VLADVectors.shape)
	return VLADVectors

#returns a dictionary that maps the path to a label for the annotations text file
def getAllLabels(labelPath):
	f = open(labelPath, "r")
	content = [x.strip('\n') for x in f.readlines()]
	f.close()
	pathToLabel ={}
	for line in content:
		pAndL = line.split()
		path = pAndL[0][:-4] + ".npy"
		label = pAndL[1]
		#print ("path " + path)
		#print ("label " + label)
		pathToLabel[path] = label
	return pathToLabel

def readFiletoNPArray(filePath):
	arr = np.load(filePath)
	return arr

#flatten and normalize the MFCC
def flattenMFCC(MFCC):
	flattened = []
	for window in MFCC:
		window.flatten()
		sum = 0
		for val in window:
			sum += val
		flattened.append(sum)
	return normalize(np.array(flattened))

# center data and set within range [-1, +1]
def normalize(data):
    preprocessing.scale(data, 0, True, False, False)
    preprocessing.MinMaxScaler((-1, 1), False).fit_transform(data)
    return data

def pipeline(labelPath, fileDir, numCentroids = 50, batchSize = 100, num_iter = 1000):
	labels = getAllLabels(labelPath)
	MFCCs = []
	includedLabels = []
	for path, label in labels.iteritems():
		#print(fileDir + path)
		arrPath = fileDir + path
		try:
			arr = flattenMFCC(readFiletoNPArray(arrPath))
			MFCCs.append(arr)
			includedLabels.append(label)
		except:
			print("couldn't find " + arrPath)

	#run KMeans on 10% of values
	print(len(MFCCs))
	print(len(includedLabels))
	KMeansVals = shuffle (MFCCs, n_samples = int(math.ceil(len(MFCCs)/10)), random_state=5261986)
	#print(len(KMeansVals))
	#print("KMeansVals")
	#print(KMeansVals)
	centroids = onlineKMeans(numCentroids, batchSize, num_iter, KMeansVals)
	VLADVectors = VLAD(centroids, MFCCs)
	print(len(VLADVectors))
	# for vector in VLADVectors:
	# 	print (vector)
	# 	print(vector.shape)
	train_target, test_target, train_data, test_data = train_test_split(includedLabels, VLADVectors, train_size=0.5, random_state=526198)
	clf = svm.LinearSVC()
	print(clf.fit(train_data, train_target))
	numRight = 0
	for index, point in enumerate(test_data):
		#print(point)
		val = clf.predict(point)
		dec = clf.decision_function(point)
		# print("decision function")
		# print(dec)
		# print("val")
		# print(val)
		# print("label")
		# print(test_target[index])
		if val == test_target[index]:
			numRight +=1
	print(numRight)
	print(len(test_data))
	return float(numRight)/float(len(test_data))


#do a grid search for best parameters
bestNum = 5
bestBatch = 10
bestVal =0

for numCentroids in range(5, 100, 5):
	for batchSize in range(50, 250, 50):
		val = pipeline("annotationstest/raw_list.txt", "data/SimpsonsNewTest/", numCentroids = numCentroids, batchSize = batchSize)
		print("current number of centroids is ")
		print(numCentroids)
		print("current batch size is ")
		print(batchSize)
		print("current score of ")
		print(val)
		if val > bestVal:
			bestVal = val
			bestNum = numCentroids
			bestBatch = batchSize

print("best number of centroids is ")
print(bestNum)
print("best batch size is ")
print(bestBatch)
print("with a best score of ")
print(bestVal)


# pipeline("annotationstrain/raw_list.txt", "data/SimpsonsNewTrain/", numCentroids = 50, batchSize = 50)

# irisKMeans = shuffle (iris.data, n_samples = int(math.ceil(len(iris.data)/10)), random_state=5261986)

# iris.data = VLAD(onlineKMeans(10, 10, 50, irisKMeans), iris.data)

# train_target, test_target, train_data, test_data = train_test_split(iris.target, iris.data, train_size=0.5, random_state=526198)
# clf = svm.LinearSVC()
# print(clf.fit(train_data, train_target))
# numRight = 0
# for index, point in enumerate(test_data):
# 	val = clf.predict(point)
# 	dec = clf.decision_function(point)
# 	print("decision function")
# 	print(dec)
# 	print("val")
# 	print(val)
# 	print("label")
# 	print(test_target[index])
# 	if val == test_target[index]:
# 		numRight +=1
# print(numRight)
# print(len(test_data))



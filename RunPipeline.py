import random
from numpy import linalg as LA
from sklearn.decomposition import PCA
import sys
import math
from sklearn.neighbors import LSHForest
from sklearn import preprocessing
import os
import time
import numpy as np
from tempfile import TemporaryFile
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import warnings
import operator
from scipy.spatial import distance

warnings.filterwarnings('ignore')

# center data and set within range [-1, +1]
def myNormalize(data):
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
		print("KMeans iteration number " + str(i))
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

def applyVLAD(point, lshf, centroids):
		# print(point)
		distances, centroidIndex = lshf.kneighbors(point, n_neighbors=len(centroids))
		p = centroidIndex[0].argsort(kind='mergesort')
		VLADVector = myNormalize(distances[0][p])
		# print(VLADVector)
		return VLADVector

#this one uses euclidean distance rather than approximate nearest neighbors
def applyVLAD2(point, centroids):
	returnArr = np.copy(centroids)
	returnArr = np.apply_along_axis(distance.euclidean, 1, returnArr, point)
	return returnArr
	# print(returnArr)
	# print(returnArr.shape)
	# return distance.cdist()

def VLADSingle(centroids, data):
	points =[]
	for point in data:
		MFCCPoint =[]
		for dim in point:
			MFCCPoint.append(dim.sum())
		MFCCPoint = myNormalize(np.array(MFCCPoint))
		print("MFCCPoint")
		print(MFCCPoint)
		newPoint =[]
		for centroid in centroids:
			newPoint.append(distance.euclidean(centroid, MFCCPoint))
		points.append(myNormalize(np.array(newPoint)))
	return points



def VLAD(centroids, data):
	if len(data) > 0:
		data = np.apply_along_axis(applyVLAD2, 1, data, centroids)
	# for idx, point in enumerate(data):
	# 	print("VLAD vector " + str(idx))		
	# 	# VLADVector =[0.0]*len(centroids)
	# 	#get distance using approximate nearest neighbors
	# 	distances, centroidIndex = lshf.kneighbors(point, n_neighbors=len(centroids))
	# 	p = centroidIndex[0].argsort(kind='mergesort')
	# 	VLADVector = myNormalize(distances[0][p])
	# 	VLADVectors.append(VLADVector)
	# 	# for index, centroid in enumerate(centroidIndex[0]):
	# 	# 	VLADVector[centroid] = distances[0][index]
	# 	# VLADVector = np.array(VLADVector)
	# 	# VLADVectors.append(VLADVector)
	# VLADVectors = np.array(VLADVectors)
	# 	#apply pca
	# pca = PCA(whiten=True)
	# VLADVectors = pca.fit_transform(VLADVectors)
	# return VLADVectors
		#pca = PCA(whiten=True)
		#data = pca.fit_transform(np.array(data))
	return data

#returns a dictionary that maps the path to a label for the annotations text file
def getAllLabels(labelPath, subset=False):
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
		if not subset or (label=="0" or label =="1"):
			pathToLabel[path] = label
	return pathToLabel

#helper function that iterates through labels and data from files and assigns each window of MFCCs to a label
def processMFCCsAndLabels(data, target):
	MFCCs =[]
	Labels =[]
	for label, sample in zip(target, data):
				arrs = flattenMFCC(sample)
				for arr in arrs:
					MFCCs.append(arr)
					Labels.append(label)
	return [MFCCs, Labels]

#similar to the above but for a single data point without the label
def processMFCCAndLabel(sample):
	MFCCs =[]
	arrs = flattenMFCC(sample)
	for arr in arrs:
		MFCCs.append(arr)
	return MFCCs

def readFiletoNPArray(filePath):
	arr = np.load(filePath)
	# print(arr)
	# print(arr.shape)
	return arr

# flatten and normalize the MFCC
def flattenMFCC(MFCC):
	# flattened = []
	# for window in MFCC:
	# 	window.flatten()
	# 	sum = 0
	# 	for val in window:
	# 		sum += val
	# 	flattened.append(sum)
	# return normalize(np.array(flattened))
	newList =[]
	for window in MFCC:
		window = np.ravel(window)
		# print("window before normalizing")
		# print(window)
		# print(window.shape)
		# newWindow = normalize(newWindow)
		# print("window after normalizing ")
		# print(newWindow)
		# newList.append(newWindow)
		window = myNormalize(window)
		newList.append(window)
	return newList

# center data and set within range [-1, +1]
def normalize(data):
    preprocessing.scale(data, 0, True, False, False)
    preprocessing.MinMaxScaler((-1, 1), False).fit_transform(data)
    return data

def pipeline(labelPath, fileDir, numCentroids = 50, C=1.0, batchSize = 100, num_iter = 1000, subset=False):
	labelsAndPaths = getAllLabels(labelPath, subset)
	labels =[]
	data =[]
	counter = 0
	print("loading labels and paths")
	t0 = time.time()
	# extract labels and data
	for path, label in labelsAndPaths.iteritems():
		arrPath = fileDir + path
		try:
			arr = readFiletoNPArray(arrPath)
			data.append(arr)
			labels.append(label)
			counter += 1
			print("loaded " + arrPath)
		except:
			print("couldn't find " + arrPath)
	print("was able to read " + str(counter) + " files")
	print( "took " + str(time.time() - t0) + " seconds to run")

	print("splitting train and test data")
	# split up train and test data
	t0 = time.time()
	train_target, test_target, train_data, test_data = train_test_split(labels, data, train_size=0.30, random_state=526198)
	train_target, test_target, train_data, test_data = train_test_split(train_target, train_data, train_size=0.5, random_state=526198)
	#train_target, test_target, train_data, test_data = train_test_split(labels, data, train_size=0.5, random_state=526198)
	print( "took " + str(time.time() - t0) + " seconds to run")


	#process the train data
	print("flattening MFCCs")
	t0 = time.time()
	MFCCs, MFCCLabels = processMFCCsAndLabels(train_data, train_target)
	print( "took " + str(time.time() - t0) + " seconds to run")
	print("total number of MFCCs " + str(len(MFCCs)))
	#run KMeans on 10% of values
	KMeansVals = shuffle (MFCCs, n_samples = int(math.ceil(len(MFCCs)/10)), random_state=5261986)
	print("running KMeans")
	t0 = time.time()
	centroids = onlineKMeans(numCentroids, batchSize, num_iter, KMeansVals)
	print( "took " + str(time.time() - t0) + " seconds to run")
	print("creating VLAD Vectors")
	t0 = time.time()
	VLADVectors = VLAD(centroids, MFCCs)
	print( "took " + str(time.time() - t0) + " seconds to run")

	print("training our SVM")
	t0 = time.time()
	clf = svm.SVC(kernel ='linear', C=C)
	print(clf.fit(VLADVectors, MFCCLabels))
	print( "took " + str(time.time() - t0) + " seconds to run")
	t0 = time.time()
	outcome = voteOnLabels(clf, test_data, test_target, centroids)
	print( "took " + str(time.time() - t0) + " seconds to run")
	f  = open('/home/ma2510/machine_learning/runLog.txt', 'a')
	writeString = str(C) + "," + str(batchSize) + "," + str(numCentroids) + "," + str(outcome) +"\n"
	print(writeString)
	f.write(writeString)
	f.close()


# def pipeline2(labelPath, fileDirTrain, fileDirTest, C=1.0, numCentroids = 50, batchSize = 100, num_iter = 1000, subset=False):
# 	labels =[]
# 	paths = []
# 	for p, l in getAllLabels(labelPath, subset).iteritems():
# 		labels.append(l)
# 		paths.append(p)
# 	# labelsTrain, labelsTest, pathsTrain, pathsTest = train_test_split(labels, paths, train_size=0.05, random_state=5261986)
# 	labelsTrain, labelsTest, pathsTrain, pathsTest = train_test_split(labels, paths, train_size=0.5, random_state=5261986)
# 	labelsForData =[]
# 	data =[]
# 	counter = 0
# 	print("loading labels and paths for train data")
# 	t0 = time.time()
# 	# extract labels and data
# 	for path, label in zip(pathsTrain, labelsTrain):
# 		arrPath = fileDirTrain + path
# 		try:
# 			arr = readFiletoNPArray(arrPath)
# 			data.append(np.array(arr))
# 			labelsForData.append(label)
# 			counter += 1
# 			print("loaded " + arrPath)
# 		except:
# 			print("couldn't find " + arrPath)
# 	print("was able to read " + str(counter) + " files")
# 	print( "took " + str(time.time() - t0) + " seconds to run")
# 	#process the train data
# 	print("flattening MFCCs")
# 	t0 = time.time()
# 	MFCCs, MFCCLabels = processMFCCsAndLabels(data, labelsForData)
# 	print( "took " + str(time.time() - t0) + " seconds to run")
# 	print("total number of MFCCs " + str(len(MFCCs)))
# 	#run KMeans on 10% of values
# 	KMeansVals = shuffle (MFCCs, n_samples = int(math.ceil(len(MFCCs)/10)), random_state=5261986)
# 	print("running KMeans")
# 	t0 = time.time()
# 	centroids = onlineKMeans(numCentroids, batchSize, num_iter, KMeansVals)
# 	print( "took " + str(time.time() - t0) + " seconds to run")
# 	print("creating VLAD Vectors")
# 	t0 = time.time()
# 	VLADVectors = VLAD(centroids, np.array(MFCCs))
# 	print( "took " + str(time.time() - t0) + " seconds to run")

# 	print("training our SVM")
# 	t0 = time.time()
# 	clf = svm.LinearSVC()
# 	print( "took " + str(time.time() - t0) + " seconds to run")
# 	print(clf.fit(VLADVectors, MFCCLabels))

# 	t0 = time.time()
# 	print("loading our test data")
# 	t0 = time.time()
# 	dataTest =[]
# 	labelsForTest =[]
# 	counter = 0
# 	for path, label in zip(pathsTest, labelsTest):
# 		arrPath = fileDirTest + path
# 		try:
# 			arr = myNormalize(readFiletoNPArray(arrPath))
# 			dataTest.append(arr)
# 			labelsForTest.append(label)
# 			counter += 1
# 			print("loaded " + arrPath)
# 		except:
# 			print("couldn't find " + arrPath)
# 	print("was able to read " + str(counter) + " files")
# 	print( "took " + str(time.time() - t0) + " seconds to run")
# 	print("applying VLAD")
# 	t0 = time.time()
# 	testVLADVectors = VLADSingle(centroids, dataTest)
# 	print( "took " + str(time.time() - t0) + " seconds to run")
# 	print("scoring test data")
# 	print("score is ")
# 	t0 = time.time()
# 	print(clf.score(testVLADVectors, labelsForTest))
# 	print( "took " + str(time.time() - t0) + " seconds to run")





def classifyVLADWindows(clf, test_data, test_target, centroids):
		#process test data
	testMFCCs, testLabels = processMFCCsAndLabels(test_data, test_target)
	print("creating testVLAD vectors")
	t0 = time.time()
	testVLADVectors = VLAD(centroids, testMFCCs)
	print( "took " + str(time.time() - t0) + " seconds to run")
	print("scoring test data")
	print("score is ")
	t0 = time.time()
	print(clf.score(testVLADVectors, labelsTest))
	print( "took " + str(time.time() - t0) + " seconds to run")

def voteOnLabels(clf, data, target, centroids):
	numRight = 0
	for data2, target in zip(data, target):
		testMFCCs = processMFCCAndLabel(data2)
		testVLADVectors = VLAD(centroids, testMFCCs)
		classifiers = {}
		for vector in testVLADVectors:
			if vector.shape[0] == len(centroids):
				val = clf.predict(vector)
				val = val[0]
				if val in classifiers:
					classifiers[val] = classifiers[val]+1;
				else:
					classifiers[val] = 1;
		bestSize =0
		bestKey = -1
		for key in classifiers:
			if(classifiers[key] > bestSize):
				bestSize = classifiers[key]
				bestKey = key

		# print("best Key")
		# print(bestKey)
		# print("target")
		# print(target)
		if bestKey == target:
			numRight += 1
	print(numRight)
	print(len(data))
	print float(numRight)/float(len(data))
	return float(numRight)/float(len(data))






# def pipeline(labelPath, fileDir, numCentroids = 50, batchSize = 100, num_iter = 1000, subset=False):
# 	labels = getAllLabels(labelPath, subset)
# 	MFCCs = []
# 	includedLabels = []
# 	counter = 0
# 	for path, label in labels.iteritems():
# 		#print(fileDir + path)
# 		arrPath = fileDir + path
# 		try:
# 			arrs = flattenMFCC(readFiletoNPArray(arrPath))
# 			for arr in arrs:
# 				MFCCs.append(arr)
# 				includedLabels.append(label)
# 			counter += 1
# 			print("loaded " + arrPath)
# 		except:
# 			print("couldn't find " + arrPath)
# 	print("was able to read " + str(counter) + " files")

	
# 	print(len(MFCCs))
# 	print(len(includedLabels))
# 	#run KMeans on 10% of values
# 	KMeansVals = shuffle (MFCCs, n_samples = int(math.ceil(len(MFCCs)/10)), random_state=5261986)
# 	#print(len(KMeansVals))
# 	#print("KMeansVals")
# 	#print(KMeansVals)
# 	print("running KMeans")
# 	centroids = onlineKMeans(numCentroids, batchSize, num_iter, KMeansVals)
# 	print("creating VLAD Vectors")
# 	VLADVectors = VLAD(centroids, MFCCs)
# 	print(len(VLADVectors))
# 	# for vector in VLADVectors:
# 	# 	print (vector)
# 	# 	print(vector.shape)
# 	train_target, test_target, train_data, test_data = train_test_split(includedLabels, VLADVectors, train_size=0.5, random_state=526198)
# 	print("training our SVM")
# 	clf = svm.LinearSVC()
# 	print(clf.fit(train_data, train_target))
# 	numRight = 0
# 	for index, point in enumerate(test_data):
# 		#print(point)
# 		val = clf.predict(point)
# 		dec = clf.decision_function(point)
# 		# print("decision function")
# 		# print(dec)
# 		# print("val")
# 		# print(val)
# 		# print("label")
# 		# print(test_target[index])
# 		if val == test_target[index]:
# 			numRight +=1
# 	print(numRight)
# 	print(len(test_data))
# 	return float(numRight)/float(len(test_data))

# #do a grid search for best parameters
# bestNum = 5
# bestBatch = 10
# bestVal =0
pipeline("/home/ma2510/machine_learning/annotationstrain/raw_list.txt", "/home/ma2510/machine_learning/data/SimpsonsNewBetterTrain/", C=1.0, numCentroids = 512, batchSize = 500, subset = False)
# 	for batchSize in range(50, 250, 50):
# 		val = pipeline("annotationstest/raw_list.txt", "data/SimpsonsNewTest/", numCentroids = numCentroids, batchSize = batchSize)
# 		print("current number of centroids is ")
# 		print(numCentroids)
# 		print("current batch size is ")
# 		print(batchSize)
# 		print("current score of ")
# 		print(val)
# 		if val > bestVal:
# 			bestVal = val
# 			bestNum = numCentroids
# 			bestBatch = batchSize

# print("best number of centroids is ")
# print(bestNum)
# print("best batch size is ")
# print(bestBatch)
# print("with a best score of ")
# print(bestVal)


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



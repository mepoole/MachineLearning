from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import random
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
import glob
from sklearn.neighbors import KNeighborsClassifier
import wave
import librosa
from sklearn.preprocessing import normalize
from scipy.spatial import distance
import time


'''
NOTES: I've included in caps where you would need to include your own stuff. They are: kmeans algo, an objective function for
the kmeans output, and a method to get the labels for each of the files. These functions take "paths" as parameters a lot of times.
The organization of the data that I'm assuming is that there are folders "cluster" "train" and "test".
"cluster" contains a subset of the data (about 10% says Zaid) that is used to train kmeans on.
"train" would contain the training set for the nearest neighbors classifier (would need both wav files and labels)
"test" would contain the test set for the nearest neighbors classifier (would need both wav files and labels)
'''
def KMeans_plusplus (datapoints, k, max_iter):
	print("k")
	print(k)
	clusters = []
	#generated initial random centroids
	seed = 52686
	random.seed(++seed)
	centroid = random.sample(datapoints, 1)
	centroids = centroid
	for i in range(1, int(k)):
		distributionObj= createDistribution(centroids, datapoints)
		newC = randomWeightedPoint(distributionObj, ++seed)
		centroids.append(newC)
	clusters = []
	for centr in centroids:
		cluster = []
		cluster.append(centr)
		clusters.append(cluster)

	for cluster in clusters:
		cluster.append([]);
	#iterate through k-means
	for idx in range (0, max_iter):
		#reset list of points assigned
		for cluster in clusters:
			del cluster[1][:]
		#assign datapoints to indices
		for point in datapoints:
			clusterNum = 0
			dist = sys.maxint;
			for cluster in enumerate(clusters):
				theDist = distance.euclidean(cluster[1][0], point)
				if theDist < dist:
					dist = theDist
					clusterNum = cluster[0]
			clusters[clusterNum][1].append(point)
		#reassign the centroids
		for cluster in clusters:
			#print cluster
			#print "\n"
			newCentroid = []
			for dim in cluster[1][0]:
				newCentroid.append(0.0)
			for point in cluster[1]:
				for dim in enumerate(point):
					newCentroid[dim[0]] += dim[1]
			#print newCentroid
			#print len(cluster[1])
						
			if len(cluster[1]) != 0:
				newCentroid[:] = [x/len(cluster[1]) for x in newCentroid]
			cluster[0] = newCentroid
	return clusters

#returns the objective function for a kmeans clustering
def KMeansObjective(clusters):
	totalPoints = 0
	totalDist = 0;
	for cluster in clusters:
		totalPoints += len(cluster[1])
		for point in cluster[1]:
			totalDist += euclideanDist(point, cluster[0])
	return totalDist/totalPoints

#euclidean distance helper function
def euclideanDist(point1, point2):
	squareDist = 0
	for point in enumerate(point1):
		squareDist += math.pow(point[1]-point2[point[0]], 2)
	return math.sqrt(squareDist);

# #given a path to a file, returns the mfcc vector for that file
# def getMFCC(filename):
# 	y, sr = librosa.load(filename, sr=16000)
# 	return librosa.feature.mfcc(y=y, sr=sr)

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

def readFiletoNPArray(filePath):
	arr = np.load(filePath)
	# print(arr)
	# print(arr.shape)
	return arr

def learnvocabulary(arrs, clusterSize=3, iterations=50):
	MFCCs = []
	for mfcc in arrs:
		mfcc = flattenMFCC(mfcc)
		MFCCs.append(mfcc)
	return KMeans_plusplus(MFCCs, clusterSize, iterations)

#flatten and normalize the MFCC
def flattenMFCC(MFCC):
	flattened = []
	for window in MFCC:
		window.flatten()
		sum = 0
		for val in window:
			sum += val
		flattened.append(sum)
	return myNormalize(np.array(flattened))

def flattenY(MFCC):
	newArr = np.sum(np.array(MFCC), axis=0)
	return myNormalize(newArr)

# center data and set within range [-1, +1]
def myNormalize(data):
	preprocessing.scale(data, 0, True, False, False)
	preprocessing.MinMaxScaler((-1, 1), False).fit_transform(data)
	return data


#given a set of clusters and a point, returns the centroid that is closest to that point
def getCentroid(point, clusters):
	bestCluster=[]
	bestDistance = sys.maxint
	for cluster in clusters:
		theDist = distance.euclidean(cluster, point)
		if theDist < bestDistance:
			bestDistance = theDist
			bestCluster = cluster
	return bestCluster


def getbof(arr, clusters):
	mfcc = flattenMFCC(arr)
	bof =[]
	for point in mfcc:
		bof.append(getCentroid(point, clusters))
	return bof

#gets the bag of features and labels for all the wav files in the directory
def getLabelsAndBofs(arrs, target, clusters):
	bofs = []
	labels = []
	for arr, label in zip(arrs, target):
		bofs.append(getCentroid(arr, clusters))
		labels.append(label)
	return [bofs, labels]

#gets the label for the file
def getLabel(filename):
	dirs = filename.split('/')
	fileToRead = dirs[0] + '/' + dirs[1] + '/labels.txt'
	with open(fileToRead) as f:
		content = f.readlines()
		for line in content:
			labels = line.split(' ')
			if(labels[0] == dirs[-1]):
				return int(labels[1][0])

#function to do a grid search to find the best cluster size
def getBestCluster(path, minClust, maxClust):
	bestVal = KMeansObjective(learnvocabulary(path, clusterSize=minClust))
	for x in range(minClust, maxClust):
		objVal = KMeansObjective(learnvocabulary(path, clusterSize=x))
		if x < bestVal:
			bestVal = x
	return bestVal

def pipeline(labelPath, fileDir, numCentroids = 2, batchSize = 100, num_iter = 1000, subset=False, kNeighbors =5):

	labelsAndPaths = getAllLabels(labelPath, subset)

	print("loading labels and paths")
	t0 = time.time()
	counter =0
	data =[]
	labels = []
	# extract labels and data
	for path, label in labelsAndPaths.iteritems():
		arrPath = fileDir + path
		try:
			arr = flattenY(readFiletoNPArray(arrPath))
			arr = arr.reshape((20,))
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
	# train_target, test_target, train_data, test_data = train_test_split(labels, data, train_size=0.1, random_state=526198)
	# train_target, test_target, train_data, test_data = train_test_split(train_target, train_data, train_size=0.5, random_state=526198)
	train_target, test_target, train_data, test_data = train_test_split(labels, data, train_size=0.5, random_state=526198)

	# KMeansVals = shuffle (train_data, n_samples = int(math.ceil(len(train_data)/10)), random_state=5261986)

	# print( "took " + str(time.time() - t0) + " seconds to run")

	bestVal = -sys.maxint - 1
	bestK = 0
	bestClust = 0
	# for clust in (math.pow(2, x) for x in range(1, 13)):
		#first, we do a grid search to find the best clustering:
		# bestVal = getBestCluster(KMeansVals, 1, 10);
		# clusters = learnvocabulary(KMeansVals, clusterSize=clust)

		# #separate the centroids from the rest of the data
		# centroids = []
		# for c in clusters:
		# 	centroids.append(c[0])

		#next we get the bag of features, then run nearest neighbors (do a grid search on best k value)

	for k in range(1, 201):
		t0 = time.time()
		neigh = KNeighborsClassifier(n_neighbors=k)
		# train_data, train_labels = getLabelsAndBofs(train_data, train_target, centroids)
		# print(np.array(train_data).shape)
		# print(np.array(train_target).shape)
		neigh.fit(np.array(train_data), np.array(train_target))
		# test_data, test_labels = getLabelsAndBofs(test_data, test_target, centroids)
		predicted = neigh.predict(np.array(test_data))
		val = np.mean(predicted == test_target)
		print("["+str(k) +","+ str(val)+"],")
		print( "took " + str(time.time() - t0) + " seconds to run")
		if val > bestVal:
			bestVal = val
			bestK = k

	print("Best prediction for nearest neigbors is k=" + str(bestK) + " with an accuracy of " + str(bestVal))

pipeline("annotationstrain/raw_list.txt", "data/SimpsonsNewBetterTrain/")

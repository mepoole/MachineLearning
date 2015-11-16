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
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

def mykmeans(datapoints, k, max_iter):

	clusters = []
	#generated initial random centroids
	random.seed(52686)
	nums = random.sample(datapoints, k)
	for idx in range (0, k):
		cluster = []
		cluster.append(nums[idx])
		clusters.append(cluster)


	for cluster in clusters:
		cluster.append([]);
	#print clusters
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
				if euclideanDist(point, cluster[1][0]) < dist:
					dist = euclideanDist(point, cluster[1][0])
					clusterNum = cluster[0]
			clusters[clusterNum][1].append(point)
		#reassign the centroids
		for cluster in clusters:
			#print cluster
			#print "\n"
			newCentroid = [0.0, 0.0, 0.0, 0.0]
			for point in cluster[1]:
				for dim in enumerate(point):
					newCentroid[dim[0]] += dim[1]
			#print newCentroid
			#print len(cluster[1])
						
			if len(cluster[1]) != 0:
				newCentroid[:] = [x/len(cluster[1]) for x in newCentroid]
			cluster[0] = newCentroid
	return clusters

def mykmeans_multi(datapoints, k, max_iter, runs):
	seed = 52686

	bestObjective = sys.maxint
	bestClusters = []
	for sd in range(seed, seed + runs):
		clusters = []
		#generated initial random centroids
		random.seed(sd)
		nums = random.sample(datapoints, k)
		for idx in range (0, k):
			cluster = []
			cluster.append(nums[idx])
			clusters.append(cluster)


		for cluster in clusters:
			cluster.append([]);
		#print clusters
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
					if euclideanDist(point, cluster[1][0]) < dist:
						dist = euclideanDist(point, cluster[1][0])
						clusterNum = cluster[0]
				clusters[clusterNum][1].append(point)
			#reassign the centroids
			for cluster in clusters:
				#print cluster
				#print "\n"
				newCentroid = [0.0, 0.0, 0.0, 0.0]
				for point in cluster[1]:
					for dim in enumerate(point):
						newCentroid[dim[0]] += dim[1]
				#print newCentroid
				#print len(cluster[1])
							
				if len(cluster[1]) != 0:
					newCentroid[:] = [x/len(cluster[1]) for x in newCentroid]
				cluster[0] = newCentroid
		thisObjective = KMeansObjective(clusters)
		#print thisObjective
		if  thisObjective < bestObjective:
			bestObjective = thisObjective
			bestClusters = clusters
	#print bestObjective
	
	return bestClusters

#this function is used to return the datapoints for centroids overtime
def mykmeans_multi_overtime(datapoints, k, max_iter, runs):
	seed = 52686

	bestObjective = sys.maxint
	bestClusters = []
	centroidsOverTime = []
	for idx in range(0,k):
		centroidsOverTime.append([])
	for sd in range(seed, seed + runs):
		clusters = []
		#generated initial random centroids
		random.seed(sd)
		nums = random.sample(datapoints, k)
		for idx in range (0, k):
			cluster = []
			cluster.append(nums[idx])
			clusters.append(cluster)


		for cluster in clusters:
			cluster.append([]);
		#print clusters
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
					if euclideanDist(point, cluster[1][0]) < dist:
						dist = euclideanDist(point, cluster[1][0])
						clusterNum = cluster[0]
				clusters[clusterNum][1].append(point)
			#reassign the centroids
			for cluster in clusters:
				#print cluster
				#print "\n"
				newCentroid = [0.0, 0.0, 0.0, 0.0]
				for point in cluster[1]:
					for dim in enumerate(point):
						newCentroid[dim[0]] += dim[1]
				#print newCentroid
				#print len(cluster[1])
							
				if len(cluster[1]) != 0:
					newCentroid[:] = [x/len(cluster[1]) for x in newCentroid]
				cluster[0] = newCentroid
		thisObjective = KMeansObjective(clusters)
		#print thisObjective
		if  thisObjective < bestObjective:
			bestObjective = thisObjective
			bestClusters = clusters
			
			for cluster in enumerate(bestClusters):
				#print centroidsOverTime[cluster[0]]
				#print "\n"
				centroidsOverTime[cluster[0]].append(cluster[1][0])
	#print bestObjective
	return centroidsOverTime

def KMeans_plusplus (datapoints, k, max_iter):
	clusters = []
	#generated initial random centroids
	seed = 52686
	random.seed(++seed)
	centroid = random.sample(datapoints, 1)
	centroids = centroid
	for i in range(1, k):
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
				if euclideanDist(point, cluster[1][0]) < dist:
					dist = euclideanDist(point, cluster[1][0])
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

#helper function for kmeans_plusplus
def createDistribution(currentCentroids, points):
	totalDist = 0
	distribution = []
	probSum = 0
	for point in points:
		minDist = sys.maxint
		#print currentCentroids
		for centroid in currentCentroids:
			dist = math.pow(euclideanDist(centroid, point), 2)
			if dist < minDist:
				minDist = dist
		distribution.append([point, minDist])
		totalDist += minDist
	for pd in distribution:
		pd[1] = pd[1]/totalDist
		probSum += pd[1]
	return [probSum, distribution]

#helper function for kmeans_plusplus
# code idea from here: http://stackoverflow.com/questions/1761626/weighted-random-numbers
def randomWeightedPoint(distributionObj, seed):
	sumOfWeighted = distributionObj[0]
	distribution = distributionObj[1]
	random.seed(seed)
	rnd = random.uniform(0, sumOfWeighted)
	for i in range(0, len(distribution)):
		if rnd < distribution[i][1]:
			return distribution[i][0]
		rnd -= distribution[i][1]
	
#returns the objective function for a kmeans clustering
def KMeansObjective(clusters):
	totalPoints = 0
	totalDist = 0;
	for cluster in clusters:
		totalPoints += len(cluster[1])
		for point in cluster[1]:
			totalDist += euclideanDist(point, cluster[0])
	return totalDist/totalPoints

#prints the graph for the clustering output
def printGraph(clusters, x_index, y_index):
	colorArray = ["red", "orange", "yellow", "green", "blue", "purple", "cyan", "magenta", "black", "white"]
	for cluster in enumerate(clusters):
		if len(cluster[1][1]) >0:
			arr = np.array(cluster[1][1])
			plt.scatter(arr[:, x_index], arr[:, y_index], c=colorArray[cluster[0]])

#prints the movement of centroids
def printOvertime(centroids, x_index, y_index):
	colorArray = ["red", "orange", "yellow", "green", "blue", "purple", "cyan", "magenta", "black", "white"]
	#print centroids
	for centroid in enumerate(centroids):
			arr = np.array(centroid[1])
			plt.plot(arr[:, x_index], arr[:, y_index],  marker='o', c=colorArray[centroid[0]])

#euclidean distance helper function
def euclideanDist(point1, point2):
	squareDist = 0
	for point in enumerate(point1):
		squareDist += math.pow(point[1]-point2[point[0]], 2)
	return math.sqrt(squareDist)

#improved euclidean distance function
def betterEuclideanDist(point1, point2):
	return math.sqrt(np.dot(point1, point1) - 2*(np.dot(point1, point2)) + np.dot(point2, point2))

#KMeans++ with the improved distance function
def betterKMeans_plusplus(datapoints, k, max_iter):
	clusters = []
	#generated initial random centroids
	seed = 52686
	random.seed(++seed)
	centroid = random.sample(datapoints, 1)
	centroids = centroid
	for i in range(1, k):
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
				newDist = betterEuclideanDist(point, cluster[1][0]) 
				if newDist < dist:
					dist = newDist
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

#given a path to a file, returns the mfcc vector for that file
def getMFCC(filename):
	y, sr = librosa.load(filename, sr=16000)
	return librosa.feature.mfcc(y=y, sr=sr)

def learnvocabulary(path, clusterSize=3, iterations=50):
	path +='/*.wav'
	MFCCs = []
	for filename in glob.glob(path):
		MFCCs.extend(getMFCC(filename))
	return KMeans_plusplus(flattenMFCC(MFCCs), clusterSize, iterations)

#flatten and normalize the MFCC
def flattenMFCC(MFCCs):
	flattened = []
	for MFCC in MFCCs:
		MFCC.flatten()
		sum = 0
		for point in MFCC:
			sum += point
		flattened.append(sum)
	return normalize(np.array(flattened))


# center data and set within range [-1, +1]
def normalize(data):
	preprocessing.scale(data, 0, True, False, False)
	preprocessing.MinMaxScaler((-1, 1), False).fit_transform(data)
	return data


#given a set of clusters and a point, returns the centroid that is closest to that point
def getCentroid(clusters, point):
	bestCluster=[]
	bestDistance = sys.maxint
	for cluster in clusters:
		if euclideanDist(cluster, point) < bestDistance:
			bestDistance = euclideanDist(cluster, point)
			bestCluster = bestCluster
	return bestCluster


def getbof(file, clusters):
	mfcc = flattenMFCC(getMFCC(file))[0]
	bof =[]
	for point in mfcc:
		bof.append(getCentroid(mfcc, clusters))
	return bof

#gets the bag of features and labels for all the wav files in the directory
def getLabelsAndBofs(path, clusters):
	path +='/*.wav'
	bofs = []
	labels = []
	for filename in glob.glob(path):
		bofs.append(getbof(filename))
		labels.append(getLabel(filename))
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

#first, we do a grid search to find the best clustering:
bestVal = getBestCluster('Simpsons/cluster', 1, 10);
clusters = learnvocabulary('Simpsons/cluster', clusterSize=bestVal)

#separate the centroids from the rest of the data
centroids = []
for c in clusters:
	centroids.append(c[0])

#next we get the bag of features, then run nearest neighbors (do a grid search on best k value)
bestVal = sys.minint
bestK = 0
for k in range(1, 10):
	neigh = KNeighborsClassifier(n_neighbors=k)
	train_data, train_labels = getLabelsAndBofs('Simpsons/train', centroids)
	neigh.fit(train_data, train_labels)
	test_data, test_labels = getLabelsAndBofs('Simpsons/test', centroids)
	predicted = neigh.predict(test_data)
	val = np.mean(predicted == test_labels)
	if val > bestVal:
		bestVal = val
		bestK = k

print("Best prediction for nearest neigbors is " + bestK + " with an accuracy of " + bestVal)

#function that generates tuples for the two possible attributes
def ParamGrid(attributes):
	returnLists = []
	for att0 in attributes[0]:
		for att1 in attributes[1]:
			returnLists.append([att0, att1)

#function that runs the algorithm for a given tuple of parameters of kMeans and NearestNeighbors
def RunForTheseParams(params):
	clusterSize = params[0]
	kNeighbors = params[1]
	clusters = learnvocabulary('Simpsons/cluster', clusterSize=bestVal)

	#separate the centroids from the rest of the data
	centroids = []
	for c in clusters:
		centroids.append(c[0])
	neigh = KNeighborsClassifier(n_neighbors=kNeighbors)
	train_data, train_labels = getLabelsAndBofs('Simpsons/train', centroids)
	neigh.fit(train_data, train_labels)
	test_data, test_labels = getLabelsAndBofs('Simpsons/test', centroids)
	predicted = neigh.predict(test_data)
	return np.mean(predicted == test_labels)

searchValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

valuesToSearch = paramGrid([searchValues, searchValues])

bestParams =[]
bestAccuracy = 0

#I'm running in a for loop but these function calls could be easily be divided up and multithreaded
for value in valuesToSearch:
	accuracy = RunForTheseParams(value)
	if accuracy > bestAccuracy:
		bestAccuracy = accuracy
		bestParams = value

print("Best KMeans Cluster size is: " + bestParams[0] + "Best prediction for nearest neigbors is " + bestParams[1] + " with an accuracy of " + bestVal)

#example 2 code


categories = ["pos", "neg"]
reviews = load_files("text_analytics/data/movie_reviews/txt_sentoken", categories=categories)
data_train, data_test, labels_train, labels_test = train_test_split(reviews.data, reviews.target, random_state=0)

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', KNeighborsClassifier())])

text_clf=text_clf.fit(data_train, labels_train) 
predicted = text_clf.predict(data_test)
np.mean(predicted == labels_test)

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False), 'clf__n_neighbors': (2, 10)}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(data_train[:300], labels_train[:300])

best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
	print("%s: %r" % (param_name, best_parameters[param_name]))
score
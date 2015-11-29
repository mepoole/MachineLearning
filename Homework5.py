import math
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from matplotlib import colors
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

def compute_obj(C, h, w, points, labels):
	norm = math.pow(np.linalg.norm(w),2)
	summation =0
	for idx, point in enumerate(points):
		summation +=huber_hinge(labels[idx], f(w, point), h)
	return (summation/float(len(points)) * C) + norm


def hinge(y, t):
	return max(0, 1-y*t)

def huber_hinge(y, t, h):
	if y*t > 1+h:
		return 0.
	elif abs(1-y*t) <= h:
		numerator = math.pow((1+h-y*t), 2)
		denominator = 4.*h
		return numerator/denominator
	else:
		return 1-y*t

def misclassificationError(y, t):
	if y*t <= 0 :
		return 1
	else:
		return 0

def classifier(w, x):
	return 1 if f(w, x) > 0 else -1

def f(w, x):
	return np.dot(w, x)

	#removes any classifiers that aren't 0 or 1 from sorted dataset
def max2Classifiers (target, data):
	#get index for where classifier 2 starts
	startidx= 0
	for idx, val in enumerate(iris.target):
		if target[idx] == 2:
			startidx = idx;
			break
	endidx = target.size

	#remove classifier 2, making the dataset binary
	target = np.delete(target, np.arange(startidx, endidx), None)
	data = np.delete(data, np.arange(startidx, endidx), 0)
	return [target, data]

#adds a dimension to our set of points
#def addDimension(points):


#hacky thing to deal with labels not being 1 or -1
def convertLabels(labels):
	for idx, label in enumerate(labels):
		if label==0:
			labels[idx] =  -1
	return labels

#add additional dimension to datapoints
def addIntercept(points):
	newPoints = []
	for point in points:
		newPoints.append(np.append(point, [1.0]))
	return np.array(newPoints)

#prints a graph of the misclassification error, hinge loss and huber hinge loss
def printGraph():
	yt = np.arange(-10, 10, 0.1)
	t = np.arange(0, 10, 0.1)
	hingeOutput =[]
	huber_hingeOutput =[]
	misclassificationOutput =[]
	for val in reversed(t):
		hingeOutput.append(hinge(-1, val))
		huber_hingeOutput.append(huber_hinge(-1, val, 4))
		misclassificationOutput.append(misclassificationError(-1, val))
	for val in t:
		hingeOutput.append(hinge(1, val))
		huber_hingeOutput.append(huber_hinge(1, val, 4))
		misclassificationOutput.append(misclassificationError(1, val))
	splot = plt.figure(7)
	plt.plot(yt, hingeOutput)
	plt.plot(yt, huber_hingeOutput)
	plt.plot(yt, misclassificationOutput)
	plt.title("Hinge Loss, Huber Hinge Loss, and Misclassification Error")

def compute_grad(C, h, w, points, labels):
	gradient = 2*w
	for index, point in enumerate(points):
		t = f(w, point)
		y = labels[index]
		if abs(1 -y*t) <= h:
			numerator = (C * (-y * point) * (1+h - y*t))
			denominator = (2.0 * h * len(points))
			gradient = gradient + numerator / denominator
		elif y*t < 1-h :
			numerator = (C * -y * point)
			denominator = float(len(points))
			gradient = gradient + numerator / denominator
	return gradient

def compute_grad_point(C, h, w, point, label):
	gradient = 2*w
	t = f(w, point)
	y = label
	if abs(1 -y*t) <= h:
		numerator = (C * (-y * point) * (1+h - y*t))
		denominator = (2.0 * h)
		gradient = gradient + numerator / denominator
	elif y*t < 1-h :
		numerator = (C * -y * point)
		gradient = gradient + numerator
	return gradient

def grad_checker(C, h, w, points, labels, delta, epsilon):
	numerator = compute_obj(C, h, w + epsilon * delta, points, labels) - compute_obj(C, h, w - epsilon * delta, points, labels)
	denominator = 2.0 * epsilon
	return numerator/denominator

def my_gradient_descent(C, h, points, labels, max_iter, eta, cg, co):
	w = np.zeros(points[0].shape, dtype=float)
	for i in range(0, max_iter):
		print(w)
		grad = cg(C, h, w, points, labels)
		w = w - eta* grad
	return w

#version of gradient descent to use for plotting figures on objective function and misclassification
def my_gd_per_iter(C, h, points, labels, max_iter, eta, cg, co, testPoints, testLabels):
	w = np.zeros(points[0].shape, dtype=float)
	iters = []
	misclassErrorTrain = []
	misclassErrorTest =[]
	objFunc = []
	Ws = []
	for i in range(0, max_iter):
		iters.append(i)
		misclass = totalMisclassificationError(points, labels, w)
		misclassErrorTrain.append(misclass)
		misclassTest = totalMisclassificationError(testPoints, testLabels, w)
		misclassErrorTest.append(misclassTest)
		Ws.append(w)
		objFunc.append(compute_obj(C, h, w, points, labels))
		grad = cg(C, h, w, points, labels)
		w = w - eta* grad
	return [iters, misclassErrorTrain, misclassErrorTest, objFunc, Ws]

def optimized_gradient_descent(C, h, points, labels, cg, co, max_iter = 1000, optimizationCriteria = False, performanceCriteria = False):
	w = np.zeros(points[0].shape, dtype=float)
	#create a queue for testing history of performance
	queue = []
	iter =0
	error = 100
	#create a subsample for testing misclassification error
	subsample_target, subsample_data= shuffle (labels, points, n_samples = 100, random_state=5261986)
	for i in range(0, max_iter):
		error = totalMisclassificationError(subsample_data, subsample_target, w)
		iter =i
		grad = cg(C, h, w, points, labels)
		if optimizationCriteria:
			if checkOptimizationCriteria(grad, 0.01):
				print("breaking because of optimization")
				break
		if performanceCriteria:
			print("error")
			print(error)
			if len(queue) == 10:
			 	queue.pop(0)
			 	print("error")
			 	print(error)
			 	if checkPerformanceCriteria(queue, error, 0.9):
			 		print("breaking because of performance")
			 		break
			queue.append(error)
		w = w - getBestEta(C, h, points, labels, w, grad, co)* grad
	return [w, iter, error]

def my_sgd (C, h, points, labels, cg, co, max_iter = 1000, performanceCriteria = False):
	w = np.zeros(points[0].shape, dtype=float)
	#create a queue for testing history of performance
	queue = []
	#create a subsample for testing misclassification error
	subsample_target, subsample_data= shuffle (labels, points, n_samples = 100, random_state=5261986)
	for i in range(0, max_iter):
		labels, points = shuffle (labels, points)
		for idx, point in enumerate(points):
			grad = cg(C, h, w, point, labels[idx])
			if performanceCriteria:
				error = totalMisclassificationError(subsample_data, subsample_target, w)
				print("error")
				print(error)
				if len(queue) == 10:
				 	queue.pop(0)
				 	if checkPerformanceCriteria(queue, error, 0.9):
				 		print("breaking because of performance")
				 		return w
				queue.append(error)
			w = w - getBestEta(C, h, points, labels, w, grad, co)* grad
	return w

#used for plotting values per epoch
def my_sgd_per_epoch (C, h, points, labels, cg, co, max_iter = 1000, performanceCriteria = False):
	w = np.zeros(points[0].shape, dtype=float)
	#create a queue for testing history of performance
	queue = []
	#create a subsample for testing misclassification error
	subsample_target, subsample_data= shuffle (labels, points, n_samples = 100, random_state=5261986)
	objFunc=[]
	errorPerEpoch =[]
	for i in range(0, max_iter):
		print(i)
		objFunc.append(compute_obj(C, h, w, points, labels))
		errorPerEpoch.append(totalMisclassificationError(points, labels, w))
		labels, points = shuffle (labels, points)
		for idx, point in enumerate(points):
			grad = cg(C, h, w, point, labels[idx])
			if performanceCriteria:
				error = totalMisclassificationError(subsample_data, subsample_target, w)
				print("error")
				print(error)
				if len(queue) == 10:
				 	queue.pop(0)
				 	print("error")
				 	print(error)
				 	if checkPerformanceCriteria(queue, error, 0.9):
				 		print("breaking because of performance")
				 		return w
				queue.append(error)
			w = w - getBestEta(C, h, points, labels, w, grad, co)* grad
	return [w, objFunc, errorPerEpoch]


#implementation of backtracking line search
def getBestEta(C, h, points, labels, w, grad, co):
	best = 0.1
	for k in range(0, 10):
		val = best * math.pow(1.1, k)
		if co(C, h, w- best * grad, points, labels) > co(C, h, w-val*grad, points, labels):
			best = val
		else:
			break
	return best

# returns true if the optimization criteria has been met or exceeded, false if not
def checkOptimizationCriteria(grad, epsilon):
	print(np.linalg.norm(grad))
	if np.linalg.norm(grad) <= epsilon:
		return True
	else:
		return False

def totalMisclassificationError(points, labels, w):
	total =0
	for idx, point in enumerate(points):
		total += misclassificationError(f(point, w), labels[idx])
	return total

#returns true if the performance criteria has been met or exceeded, false if not
def checkPerformanceCriteria(queue, error, rho):
	mini =0
	if len(queue) == 0:
		mini = sys.maxint
	else:
		mini = min(queue)
	print("mini is")
	print(mini)
	if error > rho * mini:
		return True
	else:
		return False

# generate datasets
#from here http://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html#example-classification-plot-lda-qda-py
def dataset_fixed_cov():
    '''Generate 2 Gaussians samples with the same covariance matrix'''
    n, dim = 500, 3
    np.random.seed(52686)
    C = np.array([[0., -0.23, .2], [-.22, 0.83, .23], [0.3, -.4, -.32]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([0, 0, 0])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y

def dataset_cov():
	'''Generate 2 Gaussians samples with different covariance matrices'''
	n, dim = 500, 2
	np.random.seed(0)
	C = np.array([[0., -1.], [2.5, .7]]) * 2.
	X = np.r_[np.dot(np.random.randn(n, dim), C), np.dot(np.random.randn(n, dim), C.T) + np.array([-4, 4])]
	y = np.hstack((np.zeros(n), np.ones(n)))
	return X, y

def random_dataset(n, dim):
	np.random.seed(0)
	C = np.random.rand(dim, dim)
	X = np.r_[np.dot(np.random.randn(n, dim), C), np.dot(np.random.randn(n, dim), C.T) + np.random.rand(1, dim)]
	y = np.hstack((np.zeros(n), np.ones(n)))
	return X, y

# center data and set within range [-1, +1]
def normalize(data):
	preprocessing.scale(data, 0, True, False, False)
	preprocessing.MinMaxScaler((-1, 1), False).fit_transform(data)
	return data

def my_svm(C, h, points, labels, cg, co, max_iter=1000, optimizationCriteria = False, performanceCriteria = False):
	train_target, test_target, train_data, test_data = train_test_split(labels, points, train_size=0.5, random_state=5261986)
	test_data = normalize(test_data)
	train_data = normalize(train_data)
	w, iter, error = optimized_gradient_descent(C, h, train_data, train_target, cg, co, max_iter, optimizationCriteria, performanceCriteria)
	totalCorrect = 0
	totalIncorrect = 0
	for idx, point in enumerate(test_data):
		if classifier(w, point) == test_target[idx]:
			totalCorrect += 1
	return [w, totalCorrect/float(len(test_data))]

# def optimized_svm(C, h, points, labels, max_iter, eta, cg, co):
# 	train_target, test_target, train_data, test_data = train_test_split(labels, points, train_size=0.5, random_state=5261986)
# 	test_data = normalize(test_data)
# 	train_data = normalize(train_data)

#function to check which step size is best
def checkStepSizes():
	vals=[]
	Ws = []
	for k in range(0, 9):
		val = 1 - 0.1 * k
		print(val)
		w = my_svm(1, 17, train_data, train_target, 1000, val, compute_grad, compute_obj)[0]
		vals.append(val)
		Ws.append(w)
	return [vals, w]


#plot functions
def plot_data(X, y, fig_index, w, title):
    splot = plt.figure(fig_index)  
    plt.scatter(X[:, 0], X[:, 1], c=y)
    x_w, y_w = w_for_graph(w)
    plt.plot(x_w, y_w)
    plt.plot()
    plt.title(title)

#plots the change in w over iterations 
def plot_w(Ws):
    splot = plt.figure(3)  
    for w in Ws:
	    x_w, y_w = w_for_graph(w)
	    plt.plot(x_w, y_w)
    plt.plot()
    plt.title("illustration of the movement of w during 1000 iterations")

def objFunc_vs_Iteration(objFunc, Iter):
	splot = plt.figure(4)
	plt.plot(objFunc, Iter)
	plt.title("Objective Function vs Iterations")

def misClassTrain_vs_iteration(misTrain, Iter):
	splot = plt.figure(5)
	plt.plot(misTrain, Iter)
	plt.title("Misclassification on Training Set vs Iterations")

def misClassTest_vs_iteration(misTest, Iter):
	splot = plt.figure(6)
	plt.plot(misTest, Iter)
	plt.title("Misclassification on Test Set vs Iterations")

#takes w and generates x and y output for graphing
def w_for_graph(w):
	x = np.arange(-10, 10, 0.1)
	y =[]
	np.arange(-10, 10, 0.1)
	for i in x:
		y.append((-w[0]/w[1])*i)
	return np.array([x, y])

#plots the objective function per epoch for sgd
def objFunc_vs_epoch(objFunc):
	x = np.arange(0, len(objFunc), 1.)
	splot = plt.figure(8)
	plt.plot(x, objFunc)
	plt.title("Objective function vs epoch for SGD")

def error_vs_epoch(error):
	x = np.arange(0, len(error), 1.)
	splot = plt.figure(10)
	plt.plot(x, error)
	plt.title("Misclassification Error vs epoch for SGD")

def best_value_C(h, points, labels, cg, co, max_iter=1000, optimizationCriteria =False, performanceCriteria = False):
	bestError = sys.maxint
	bestErrorC = 0
	bestIter = sys.maxint
	bestIterC = 0
	for i in range(1, 20):
		returnStuff = optimized_gradient_descent(i, h, points, labels, cg, co, max_iter, optimizationCriteria, performanceCriteria)
		if returnStuff[1] < bestIter:
			bestIter = returnStuff[1]
			bestIterC = i
		if returnStuff[2] < bestError:
			bestError = returnStuff[2]
			bestErrorC = i
	print ("least number of iterations is ")
	print(bestIterC)
	print("with this many iterations:")
	print(bestIter)
	print("smallest misclassificationError is ")
	print(bestErrorC)
	print("with this many misclassifications")
	print(bestError)


#my_svm(1, 17, train_data, train_target, 100, 0.001, compute_grad, compute_obj)
printGraph()

#implement line search

#implement change in the gradient



#print(compute_obj(1, 17, np.array([0,0]),  np.array([[1,1],[0,0]]), [1,1 ]))
#print(compute_grad(1, 17, np.array([2.,3.]),  np.array([[1,1],[0,0]]), [1,1 ]))
#print(grad_checker(1, 17, np.array([2.,3.]), np.array([[1,1],[0,0]]), [1,1 ], np.array([1,0]), 0.0001))
#print(grad_checker(1, 17, np.array([2.,3.]), np.array([[1,1],[0,0]]), [1,1 ], np.array([0,1]), 0.0001))
#print(my_gradient_descent(1, 17, np.array([[1,1],[0,0]]), [1,1 ], 20, 0.0001))
#print(my_gradient_descent(1, 17, iris.data, iris.target, 1000, 0.001, compute_grad, compute_obj))
#data, labels = dataset_fixed_cov()
#labels = convertLabels(labels)
#data = addIntercept(data)
#print(optimized_gradient_descent(1, 17, train_data, train_target, compute_grad, compute_obj, optimizationCriteria = True, performanceCriteria=True))
#print(my_sgd(1, 17, train_data, train_target, compute_grad_point, compute_obj, performanceCriteria=True))
#print(my_gradient_descent(1, 17, train_data, train_target, 1000, 0.001, compute_grad, compute_obj))
data, labels = random_dataset(500, 2)
# iris = load_iris()
# labels, data = max2Classifiers(iris.target, iris.data)
# data = normalize(data)
# data = addIntercept(data)
train_target, test_target, train_data, test_data = train_test_split(labels, data, train_size=0.5, random_state=5261986)
train_data = normalize(train_data)
test_data = normalize(test_data)
train_target = convertLabels(train_target)
test_target = convertLabels(test_target)
#my_sgd (25, 25, train_data, train_target, compute_grad_point, compute_obj, max_iter =1)
# my_gradient_descent(25, 25, train_data, train_target, 1, 0.001, compute_grad, compute_obj)
# output = my_sgd_per_epoch (25, 25, train_data, train_target, compute_grad_point, compute_obj, max_iter =50)
# print(output)
# objFunc_vs_epoch(output[1])
# error_vs_epoch(output[2])
# plot_data(train_data, train_target, 9, output[0], "Illustration of the w classifier")
#print(my_gd_per_iter(25, 25, train_data, train_target, 1000, 0.001, compute_grad, compute_obj, test_data, test_target))
# w = my_sgd(1, 17, train_data, train_target, compute_grad_point, compute_obj, performanceCriteria=True)
# w = optimized_gradient_descent(1, 17, train_data, train_target, compute_grad, compute_obj, performanceCriteria=True, optimizationCriteria=True)
# plot_data(data, labels, 2, w, "title")
# output = my_gd_per_iter(25, 25, train_data, train_target, 1000, 0.001, compute_grad, compute_obj, test_data, test_target)
# plot_data(train_data, train_target, 1, output[4][len(output[4])-1], "Illustration of the w classifier")
# plot_w(output[4])
# objFunc_vs_Iteration(output[0], output[3])
# misClassTrain_vs_iteration(output[0], output[1])
# misClassTest_vs_iteration(output[0], output[2])

best_value_C(1, train_data, train_target, compute_grad, compute_obj)



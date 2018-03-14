import tensorflow as tf
import numpy as np
import pandas as pd

from functools import reduce



class MLPRegressor:
	"""
	Class for performing regression using a neural network.

	To use it, you first pass a dictionary of hyperparameters to the constructor.
	(you can see the parameters in the __init__ method in MLPRegressor.py)

	Next you train it using the fit method which is intended to work similarly
	to sklearn models.
	(you pass in a list of inputs and a list of outputs)

	Finally after training, you can predict on new data using the predict method.
	(you pass in a list of examples)
	"""
	def init_weights(self,shape):
		"""given a shape variable from numpy, this will return a
		tensorflow variable of that shape initialized with
		random values with mean 0 and std-dev 1/sqrt(shape[0])

		Assuming the first dimension of your shape is the number of inputs to
		a layer of a neural network, this is a reasonable initialization.
		"""
		return tf.Variable(tf.random_normal(shape, stddev=1/np.sqrt(shape[0])))

	def model(self, X, weights ,biases, inputKeepProb, hiddenKeepProb):
		"""creates most of tensorflow graph and returns final layer.
		should be somewhat easy to alter for a different architecture.
		"""
		layer = tf.nn.dropout(X,inputKeepProb)
		for i in range(len(weights)-1):
			mlayer = tf.nn.sigmoid(tf.add(tf.matmul(layer,weights[i]),biases[i]))
			layer = tf.nn.dropout(mlayer,hiddenKeepProb)
		layer = tf.add(tf.matmul(layer,weights[-1]),biases[-1])
		# layer = tf.nn.sigmoid(layer)
		layer = tf.nn.softmax(layer)
		return layer


	def __init__(self,hparams):
		self.silent = hparams["silent"] if 'silent' in hparams else False
		self.earlyStopping = hparams["earlyStopping"] if 'earlyStopping' in hparams else False
		self.numEpochs = hparams["numEpochs"]
		self.batchSize = hparams["batchSize"]
		self.learningRate = hparams["learningRate"]
		self.numInputs = hparams["numInputs"]
		self.numOutputs = hparams["numOutputs"]
		self.layers = list(hparams["layers"])
		self.layers.insert(0,self.numInputs)
		self.layers.append(self.numOutputs)
		self.dropoutProbability = float(hparams["dropoutProbability"])
		self.validationPercent = float(hparams["validationPercent"])
		tf.reset_default_graph()
		self.X = tf.placeholder("float", [None, self.numInputs])
		self.Y = tf.placeholder("float", [None, self.numOutputs])

		depth = len(self.layers)
		if "weights" not in hparams or 'biases' not in hparams:
			if not self.silent:
				print("Randomly initializing weights!")
			self.weights = [self.init_weights([self.layers[i-1],self.layers[i]]) for i in range(1,depth)]
			self.biases = [self.init_weights([self.layers[i]]) for i in range(1,depth)]
		else:
			if not self.silent:
				print("Accepting custom weights!")
			weights = [m.astype("float32") for m in hparams['weights']]
			biases = [m.astype("float32") for m in hparams['biases']]
			self.weights = [tf.Variable(mat) for i,mat in enumerate(weights)]
			self.biases = [tf.Variable(vec) for i,vec in enumerate(biases)]

		self.inputKeepProb = tf.placeholder("float")
		self.hiddenKeepProb = tf.placeholder("float")

		self.pred = self.model(self.X, self.weights, self.biases, self.inputKeepProb,self.hiddenKeepProb)
		self.cost = tf.reduce_mean(tf.pow(self.pred-self.Y,2))
		self.trainCostSumm = tf.summary.scalar("TrainCost",self.cost)
		self.validationCostSumm = tf.summary.scalar("ValidationCost",self.cost)
		self.train_op = tf.train.AdamOptimizer(learning_rate = self.learningRate).minimize(self.cost)
		# self.train_op = tf.train.GradientDescentOptimizer(learning_rate = self.learningRate).minimize(self.cost)

		self.summ = tf.summary.merge_all()
		self.description = "epochs={};batchSize={};learningRate={};layers={};dropoutProbability={};".format(self.numEpochs,self.batchSize,self.learningRate,self.layers,self.dropoutProbability)
		self.writer = tf.summary.FileWriter("./tb/"+self.description)
		self.init_op = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.writer.add_graph(self.sess.graph)
		self.sess.run(self.init_op)

	def fit(self,x,y):

		if self.validationPercent > 0:
			trainFraction = 1-self.validationPercent
			trainingSize = int(len(x)*trainFraction)
			indices = np.random.choice(len(x),trainingSize,replace=False)
			x_validate,y_validate = zip(*[t for i,t in enumerate(zip(x,y)) if i not in indices])
			x_train = x[indices]
			y_train = y[indices]
		else:
			x_train,y_train = (x,y)

		costs = []
		for i in range(self.numEpochs):
			if i > 50 and self.earlyStopping and self.validationPercent > 0:
				now = costs[i-1]["validationCost"]
				previous = costs[i-50]["validationCost"]
				if (previous-now)/previous < 0: # if cost has increased
					break


			if not self.silent:
				print("epoch: {}".format(i))
			ct,cts = self.sess.run([self.cost,self.trainCostSumm],feed_dict={self.X:x_train,self.Y:y_train,self.inputKeepProb:1.0,self.hiddenKeepProb:1.0})
			self.writer.add_summary(cts,i)
			if self.validationPercent > 0:
				cv,cvs = self.sess.run([self.cost,self.validationCostSumm],feed_dict={self.X:x_validate,self.Y:y_validate,self.inputKeepProb:1.0,self.hiddenKeepProb:1.0})
				self.writer.add_summary(cvs,i)
			else:
				cv = None
			costs.append({"epoch":i,"trainCost":ct,"validationCost":cv})
			if not self.silent:
				print("cost training: {0:.6f}".format(ct))
				if cv is not None:
					print("cost validation: {0:.6f}".format(cv))

			numBatches = int(len(x)/self.batchSize)
			for j in range(numBatches):
				indices = np.random.choice(len(x_train),self.batchSize)
				xs = x_train[indices]
				ys = y_train[indices]
				settings = {self.X:xs,self.Y:ys,self.inputKeepProb:1.0,self.hiddenKeepProb:self.dropoutProbability}
				self.sess.run(self.train_op,feed_dict=settings)

		return costs

	def predict(self,x):
		results = []
		for i in range(0,len(x),10000):
			settings = {self.X:x[i:i+10000],self.inputKeepProb:1.0,self.hiddenKeepProb:1.0}
			results.append(self.sess.run(self.pred,feed_dict=settings))
		return np.concatenate(results)

	def score(self,x,y):
		results = []
		for i in range(0,len(x),10000):
			settings = {self.X:x[i:i+10000],self.Y:y[i:i+10000],self.inputKeepProb:1.0,self.hiddenKeepProb:1.0}
			results.append(self.sess.run(self.cost,feed_dict=settings))
		return 1-np.mean(results)

	def getWeights(self):
		return self.sess.run(self.weights)

	def getBiases(self):
		return self.sess.run(self.biases)

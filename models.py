import numpy as np
import pandas as pd
import timeit
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, learning_curve, ShuffleSplit
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
rand_state = 100

def run_models(models, x_train, x_test, y_train, y_test):
	# method to auomatically fit and predict model 
	for model in models:
		clf = model.fit(x_train, y_train)
		y_hat = clf.predict(x_cc_test)

		if length(y_train.unique()) == 2:
			roc_auc = metrics.roc_auc_score(y_test, y_preds)
			precision = metrics.precision_score(y_test, y_preds)
			recall = metrics.recall_score(y_test, y_preds)
			accuracy = metrics.accuracy_score(y_test, y_preds)

			print("ROC AUC: {}\nRecall: {}\nPrecision: {}\nAccuracy: {}".format(roc_auc, recall, precision, accuracy))
		else:
			accuracy_score = metrics.accuracy_score(y_zoo_test, y_zoo_preds)

			print("Accuracy: {}".format(accuracy_score))
		cnf_matrix_cc = confusion_matrix(y_test, y_preds)
		plt.figure()
		plot_confusion_matrix(cnf_matrix_cc, classes=[0,1], normalize=True,
		                      title='Normalized confusion matrix')

		plt.show()
	

def cross_validate(models, x_data, y_data):
	for model in models:
		clf = model
		scores = cross_val_score(clf, x_data, y_data, cv=5)
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def run(models):
	# extract train & test data for model(s)
	cc = pd.read_csv("data/creditcard.csv")
	y_cc = cc.iloc[:,-1:]
	x_cc = cc.iloc[:,:-1]

	x_cc_train, x_cc_test, y_cc_train, y_cc_test = train_test_split(x_cc, y_cc, random_state=rand_state)

	zoo = pd.read_csv("data/zoo.csv")
	y_zoo = zoo.iloc[:,-1:]
	x_zoo = zoo.iloc[:,1:-1]
	le = preprocessing.LabelEncoder()
	le.fit(y_zoo0)
	le.classes_
	y_zoo = le.transform(y_zoo0)
	x_zoo_train, x_zoo_test, y_zoo_train, y_zoo_test = train_test_split(x_zoo, y_zoo, random_state=
		rand_state)

	# run models for credit card data
	run_models(models, x_cc_train, x_cc_test, y_cc_train, y_cc_test)

	# run models for zoo data
	run_models(models, x_zoo_train, x_zoo_test, y_zoo_train, y_zoo_test)

models = [DecisionTreeClassifier(max_features='sqrt', min_samples_leaf=5),
		  DecisionTreeClassifier(max_features='sqrt', min_samples_leaf=5, class_weight='balanced'),
		  KNeighborsClassifier(),
		  MLPClassifier(),
		  GradientBoostingClassifier(min_samples_split=20,
		  AdaBoostClassifier(),	
		  SVC(kernel='linear'),
		  SVC(kernel='rbf')]
run(models)


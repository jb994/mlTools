import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

def loadBEN3():
	prefix = 'Hep53.4 BEN3+4 Tumors/'

	X1 = pd.read_csv(f'{prefix}Data/Ben3/Normal.csv', index_col=0).T.iloc[0:-1]
	X2 = pd.read_csv(f'{prefix}Data/Ben3/RCF.csv', index_col=0).T.iloc[0:-1]
	X = pd.concat((X1,X2))

	y = pd.read_csv(f'{prefix}Data/Ben3/TumorVolumes.csv', names=['liverVolume'], header=None, index_col=0)
	y = y[y.index.isin(X.index)]

	X = X.to_numpy()
	y = y.to_numpy()
	return X,y


def performanceScores(model, X_test, y_test):
	preds = model.predict(X_test)
	best_preds = np.asarray([np.argmax(line) for line in preds])
	print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
	print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
	print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))

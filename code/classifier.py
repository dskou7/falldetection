import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def getValScore(model, train, labels, cv, scoring=None):
  score = cross_val_score(model, train, labels, cv=cv, scoring=scoring)
  return score.mean(), score.std() * 2

# Load data set from our csv file into a pandas df
pwd = os.path.abspath(os.path.dirname(__file__))
relative_path_to_dataset = '../data/final_dataset_sg.csv'
data_path = os.path.join(pwd, relative_path_to_dataset)
data = pd.read_csv(data_path)
data = data.drop(columns=['Video'])
outfile = open(os.path.join(pwd, '../results.txt'), 'a')
outfile.write('-------- Using dataset: {0} --------\n'.format(relative_path_to_dataset))

val_acc = []
val_acc_ci = []
val_precision = []
val_precision_ci = []
val_recall = []
val_recall_ci = []
val_f1 = []
val_f1_ci = []
test_acc = []
test_report = []
for k in range(2, 11):
  for i in range(10):
    train, test = train_test_split(data, test_size=0.2, stratify=data['Label'].values)
    train_labels = train['Label'].values
    train = train.drop(columns=['Label'])
    test_labels = test['Label'].values
    test = test.drop(columns=['Label'])

    ### Validation set scores
    knn = KNeighborsClassifier(n_neighbors=3)
    # cv = StratifiedKFold(n_splits=5)
    # acc, acc_ci = getValScore(knn, train, train_labels, cv)
    # val_acc += [acc]
    # val_acc_ci += [acc_ci]
    # prec, prec_ci = getValScore(knn, train, train_labels, cv, 'precision')
    # val_precision += [prec]
    # val_precision_ci += [prec_ci]
    # recall, recall_ci = getValScore(knn, train, train_labels, cv, 'recall')
    # val_recall += [recall]
    # val_recall_ci += [recall_ci]
    # f1, f1_ci = getValScore(knn, train, train_labels, cv, 'f1')
    # val_f1 += [f1]
    # val_f1_ci += [f1_ci]

    ### Test set scores
    knn.fit(train, train_labels)
    preds = knn.predict(test)
    # print('Preds', preds)
    # print('True:', test_labels)
    accuracy = knn.score(test, test_labels)
    test_acc += [accuracy]
    report = classification_report(test_labels, preds)
    test_report += [report]
    # print(report)
    # input('pause')
    # print("Test accuracy: %0.2f" % accuracy)

  outfile.write('---KNN n_neighbors={0}---\n'.format(k))
  # outfile.write('Avg validation accuracy: %0.2f (+/- %0.2f)\n' % (np.mean(val_acc), np.mean(val_acc_ci)))
  # outfile.write('Avg validation precision: %0.2f (+/- %0.2f)\n' % (np.mean(val_precision), np.mean(val_precision_ci)))
  # outfile.write('Avg validation recall: %0.2f (+/- %0.2f)\n' % (np.mean(val_recall), np.mean(val_recall_ci)))
  # outfile.write('Avg validation f1 score: %0.2f (+/- %0.2f)\n' % (np.mean(val_f1), np.mean(val_f1_ci)))
  outfile.write('Avg test accuracy: %0.2f\n' % np.mean(test_acc))
  # TODO - get avg test precision, recall, and f1 from test_report.
  # TODO - also uncomment validation section eventually


''' Notes/Analysis
We get slightly different accuracies depending on how the data is randomly split (between 60%-93%, typically high 70s, low 80s)
We need several things: 
 - DONE a lot more data (this includes breaking down our clips into 60 frame bundles, or smaller)
 - DONE good noise filtering/smoothing techniques
 - DONE maybe removing videos with multiple subjects/occlusions
 - inspecting the "calcBiggestChange" f(x) we're using to calculate our stats. There may be a much better way to generate those features
'''

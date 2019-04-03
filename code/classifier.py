'''
Step 1: a: Loads data set from our csv file into a pandas df
        b: Partitions into training and test sets
Step 2: "Trains" KNN, and then predicts for test set.
Step 3: Important metrics gathered on KNN classifier (accuracy for now. Will want precision, recall, and probably others)
'''

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


### Step 1a: Loads data set from our csv file into a pandas df
pwd = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(pwd, "../data/labeled_augm_dataset.csv")
data = pd.read_csv(data_path)
data = data.drop(columns=['Video'])
### Step 1b: Partitions into training and test sets
# might want to use the "stratify" parameter. look it up. use random_state=1 for reproducibility
train, test = train_test_split(data, test_size=0.2)
train_labels = train['Label'].values
train = train.drop(columns=['Label'])
test_labels = test['Label'].values
test = test.drop(columns=['Label'])


### Step 2: "Trains" KNN, and then predicts for test set.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train, train_labels)
preds = knn.predict(test)
print('Preds', preds)
print('True:', test_labels)

### Step 3: Important metrics gathered on KNN classifier
accuracy = knn.score(test, test_labels)
print('Accuracy:', accuracy)



''' Notes/Analysis
We get slightly different accuracies depending on how the data is randomly split (between 60%-93%, typically high 70s, low 80s)
We need several things: 
 - IN PROGRESS a lot more data (this includes breaking down our clips into 60 frame bundles, or smaller)
 - IN PROGRESS good noise filtering/smoothing techniques
 - DONE maybe removing videos with multiple subjects/occlusions
 - inspecting the "calcBiggestChange" f(x) we're using to calculate our stats. There may be a much better way to generate those features
'''

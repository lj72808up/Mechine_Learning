# -*- coding:utf-8
from CommonData.prep_terrain_data import makeTerrainData
from CommonData.class_vis import prettyPicture, output_image

import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#TODO your code goes here!
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train,labels_train)

print clf.predict(features_test)
print clf.score(features_test,labels_test)   # 判断分类器的预测成功率  score(Test samples,True labels)

### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
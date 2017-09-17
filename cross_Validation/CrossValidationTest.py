# -*- coding:utf-8
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
features = iris.data
labels = iris.target
print features.shape, labels.shape  #(150, 4) (150,)

# 使用40%的数据进行评估
# 使用cross validation的train_test_split方法,分割数据
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=0)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape
# -*- coding:utf-8
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()  # 创建分类器
clf.fit(X, Y)  # 使用训练特征和训练标签填充它
print clf.predict([[-0.8, -1]])  # 使用分类器上的预测功能创建预测向量
# print clf.score(features_test,labels_test) #匹配度

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
print clf.predict([[-0.8, -1]])

from sklearn import svm

clf = svm.SVC()
clf.fit(X, Y)
print clf.predict([[-0.8, -1]])

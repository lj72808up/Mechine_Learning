# -*- encoding:utf-8
from sklearn import svm,grid_search,datasets
iris = datasets.load_iris()

parameters = {'kernel':('linear','rbf'),'C':[1,10]}
# 参数字典以及他们可取的值。这种情况下，他们在尝试找到 kernel和 C的最佳组合
# 这时，会自动生成一个不同（kernel、C）参数值组成的“网格”:
# ('rbf', 1)	   |    ('rbf', 10)
# ('linear', 1)    | 	('linear', 10)

svr = svm.SVC()  # 声明要进行参数选择的学习算法，此处不用对参数做任何一种尝试
clf = grid_search.GridSearchCV(svr,parameters)
clf.fit(iris.data,iris.target)
print clf.best_params_  # 获取尝试出来的最佳参数
print clf.best_estimator_
print ""


from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV

DecisionTreeRegressor()

aa = [i for i in range(1,11)]
print aa

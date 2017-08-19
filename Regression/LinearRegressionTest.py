# -*- coding:utf-8
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print reg.coef_       # 斜率
print reg.intercept_  # 截距


#　评估回归的性能指标：ｒ平方越高，性能越好，最大值为１
print reg.score(ages_test,net_worths_test)
print reg.score(ages_train,net_worths_train)

# 评估线性回归的方法：　可视化看回归直线，或者看测试点在拟合直线上的误差


# encoding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def doPCA(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    return pca


if __name__ == "__main__":
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=2)
    pca.fit(X)
    print("explained_variance_ratio_: %s"% pca.explained_variance_ratio_)  # 每个主成分的方差比
    first_pc = pca.components_[0]  # 向量表示的第一个主成分  [-0.83849224 -0.54491354]
    print ("first_pc: %s" % first_pc)
    second_pc = pca.components_[1]  # 向量表示的第二个主成分 [ 0.54491354 -0.83849224]
    print ("second_pc: %s" % second_pc)

    transformed_data = pca.transform(X)
    print transformed_data
    for ii,jj in zip(transformed_data,X):
        print (first_pc[0]*ii[0],first_pc[1]*ii[0])
        plt.scatter(first_pc[0]*ii[0],first_pc[1]*ii[0],color='r')   # 第一主成分上映射的点 (主成分与需要映射的点的点积)
        plt.scatter(second_pc[0]*ii[1],second_pc[1]*ii[1],color='c') # 第二主成分上映射的点
        plt.scatter(jj[0],jj[1],color='b')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

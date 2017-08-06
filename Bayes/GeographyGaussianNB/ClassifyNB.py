def classify(features_train, labels_train):
    #TODO your code goes here!
    from sklearn import svm
    clf = svm.SVC(kernel="linear")
    # from sklearn.naive_bayes import GaussianNB
    # clf = GaussianNB()
    clf.fit(features_train,labels_train)
    return clf
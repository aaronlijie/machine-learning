#python
#!/usr/bin/env python
#-*- coding : UTF-8 -*-

import sys
import os

#/Library/Python/2.7/site-packages

sys.path.insert(1, '/Library/Python/2.7/site-packages')
'''

'''
import sklearn
from sklearn import tree
import pydot
from sklearn.datasets import load_iris

from sklearn.externals.six import StringIO
import numpy as np



if __name__=="__main__":
    print "start"
    # recipes #1
    '''
    features = [[140,1],[130,1],[150,0],[170,0],[150,1]]
    labels = [0,0,1,1,0]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features,labels)
    print clf.predict([[180,1]])
    '''
    print os.path.abspath(np.__file__)
    # recipes #2
    
    iris = load_iris()
    
    print np.__version__
    print sklearn.__version__
    
    print iris.feature_names
    print iris.target_names
    print iris.data[0]
    print iris.target[0]
    test_idx = [0,50,100]

    # training data
    train_target = np.delete(iris.target,test_idx)
    train_data = np.delete(iris.data,test_idx,axis = 0)

    # testing data
    test_target = iris.target[test_idx]
    test_data = iris.data[test_idx]

    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data,train_target)

    print test_target

    #viz code
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,  
                        feature_names=iris.feature_names,  
                        class_names=iris.target_names,  
                        filled=True, rounded=True,  
                        impurity=True) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("iris.pdf")
    
    

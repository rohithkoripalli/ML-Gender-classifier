# import all dependencies
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn .naive_bayes import GaussianNB
# create all classifiers
clf = tree.DecisionTreeClassifier()
clf1 = svm.SVC()
clf2 = neighbors.KNeighborsClassifier()
clf3 = GaussianNB()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# train them on our data
clf  = clf.fit(X, Y)
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)

A = [[184,84,44],[198,92,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
B = ['male','male','male','female','female','female','male','male']

# start prediction
prediction  = clf.predict(A)
prediction1 = clf1.predict(A)
prediction2 = clf2.predict(A)
prediction3 = clf3.predict(A)

# The prediction scores
x1 = accuracy_score(prediction1,B)
x2 = accuracy_score(prediction2,B)
x3 = accuracy_score(prediction3,B)

# best prediction
if x1>x2 and x2>x3 :
	print ("svm",x1)
elif x2>x3 and x3>x1:
	print ("neighbors",x2)
else:
	print ("guassianNB",x3)




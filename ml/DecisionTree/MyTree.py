from sklearn.datasets import load_iris
from sklearn import tree
clf = tree.DecisionTreeClassifier()
iris = load_iris()
features = iris.feature_names
classes=iris.target_names
X=iris.data
y=iris.target
clf = clf.fit(X, y)

tree.export_graphviz(clf, out_file="iris_tree.dot",
    feature_names=features,
    class_names=classes,
    filled=True,
    proportion=False, precision=2,  rounded=True)
# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tsvg', 'iris_tree.dot', '-o', 'iris_tree.svg', '-Gdpi=600'])
# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'iris_tree.svg')















##########333



import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.DataFrame({'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast',
                                 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
                     'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild',
                              'Mild', 'Hot', 'Mild'],
                     'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
                              'No']})

data['Weather'] = data['Weather'].astype('category').cat.codes
data['Temp'] = data['Temp'].astype('category').cat.codes

X = data[['Weather', 'Temp']]
Y = data['Play']

# Create and train the first Decision Tree Classifier with entropy criterion
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X, Y)

plt.figure(figsize=(12, 4))
plot_tree(clf, filled=True, feature_names=['Weather', 'Temp'], class_names=['No', 'Yes'])
plt.show()

# Create and train the second Decision Tree Classifier with gini criterion and a maximum depth of 10
clf2 = DecisionTreeClassifier(criterion="gini", max_depth=10)
clf2.fit(X, Y)

plt.figure(figsize=(16, 16))
plot_tree(clf2, filled=True, feature_names=['Weather', 'Temp'], class_names=['No', 'Yes'])
plt.show()

# Create and train the third Decision Tree Classifier with gini criterion and a maximum depth of 2
clf3 = DecisionTreeClassifier(criterion="gini", max_depth=2)
clf3.fit(X, Y)

plt.figure(figsize=(12, 4))
plot_tree(clf3, filled=True, feature_names=['Weather', 'Temp'], class_names=['No', 'Yes'])
plt.show()
# Create and train another Decision Tree Classifier with gini criterion and a maximum depth of 10
clf4 = DecisionTreeClassifier(criterion="gini", max_depth=10)
clf4.fit(X, Y)

plt.figure(figsize=(16, 16))
plot_tree(clf4, filled=True, feature_names=['Weather', 'Temp'], class_names=['No', 'Yes'])
plt.show()

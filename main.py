import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import graphviz
import pydotplus
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
data = pd.read_csv(r'C:\Users\RIT\PycharmProjects\Experiment1\venv\Social_Network_Ads.csv')
data.head()
feature_cols = ['Age', 'EstimatedSalary']
x = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
classifier = DecisionTreeClassifier()
classifier = classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)
print('Accuracy Score:',metrics.accuracy_score(y_test,y_pred))
cm= confusion_matrix(y_test, y_pred)
x_set,y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop= x_set[:,0].max()+1,step=0.01),np.arange(start=x_set[:,1].min()-1,stop= x_set[:,1].max()+1,step= 0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(("red","green")))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(("red","green"))(i),label=j)
plt.title("Decision Tree(Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated salary")
plt.legend()
plt.show()


dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.write_png("Desicion_Tree.png"))



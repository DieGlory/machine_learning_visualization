from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

iris_data = iris.data
iris_label = iris.target
# iris_data_s = iris_data[:,2:4]

# print(iris.keys())
# print(iris.feature_names)
# print(iris_data)
# print(iris_label)

# x_train,x_test,y_train,y_test = train_test_split(iris_data_s,iris_label,test_size=0.3,random_state=25)

# sgd_model = SGDClassifier(alpha=0.001,max_iter=500)
# sgd_model.fit(x_train, y_train)
# sgd_model_y_pred = sgd_model.predict(x_test)
# print(classification_report(y_test,sgd_model_y_pred))

# print(np.shape(iris_data_s))

# # ax = fig.add_subplot(111,projection='3d')
# plt.scatter(iris_data_s[:,0],iris_data_s[:,1])
# plt.show()
# import numpy as np
# T0, T1 = np.meshgrid(np.linspace(-1,3,100),np.linspace(-6,2,100))
# print(T0)
# print(T1)
# x = np.linspace(0,1,40)
# X = np.vstack((np.ones(len(x)),x)).T
# print(x)

x = iris_data[:,3]
print(np.shape(x))
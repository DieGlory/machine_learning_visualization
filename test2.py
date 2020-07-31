import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris

iris = load_iris()
iris_data = iris.data
iris_label = iris.target

X = iris_data
Y = iris_label

sgd =SGDClassifier()
sgd.fit(X,Y)

a = lambda x,y: (-sgd.intercept_[0]-sgd.coef_[0][0]*x-sgd.coef_[0][1]*y) / sgd.coef_[0][2]
b = lambda x,y: (-sgd.intercept_[1]-sgd.coef_[1][0]*x-sgd.coef_[1][1]*y) / sgd.coef_[1][2]
c = lambda x,y: (-sgd.intercept_[2]-sgd.coef_[2][0]*x-sgd.coef_[2][1]*y) / sgd.coef_[2][2]

tmp = np.linspace(0,8,51)
x,y = np.meshgrid(tmp,tmp)

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, a(x,y),rstride = 1, cstride = 1, cmap = 'Reds', alpha=0.5)
ax.plot_surface(x, y, b(x,y),rstride = 1, cstride = 1, cmap = 'Greens', alpha=0.5)
ax.plot_surface(x, y, c(x,y),rstride = 1, cstride = 1, cmap = 'Blues', alpha=0.5)

ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'ob')
ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
ax.plot3D(X[Y==2,0], X[Y==2,1], X[Y==2,2],'xg')
plt.show()
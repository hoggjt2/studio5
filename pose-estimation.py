import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
data = np.loadtxt("3d_coordinates.txt", None, None, delimiter=" ", encoding="utf-16")
ID = data[:,0]
X = data[:,1]
Y = data[:,2]
Z = data[:,3]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X, Y, Z, alpha=1)
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')
def connectpoints(X,Y,Z,p1,p2):
    x1, x2 = X[p1], X[p2]
    y1, y2 = Y[p1], Y[p2]
    z1, z2 = Z[p1], Z[p2]
    ax.plot([x1,x2],[y1,y2],[z1,z2],'k-')
# lines to connect pose estimation points
connectpoints(X,Y,Z,11,12)
connectpoints(X,Y,Z,12,14)
connectpoints(X,Y,Z,14,16)
connectpoints(X,Y,Z,16,18)
connectpoints(X,Y,Z,18,20)
connectpoints(X,Y,Z,16,20)
connectpoints(X,Y,Z,16,22)
connectpoints(X,Y,Z,12,24)
connectpoints(X,Y,Z,24,26)
connectpoints(X,Y,Z,26,28)
connectpoints(X,Y,Z,28,30)
connectpoints(X,Y,Z,30,32)
connectpoints(X,Y,Z,28,32)
connectpoints(X,Y,Z,11,13)
connectpoints(X,Y,Z,13,15)
connectpoints(X,Y,Z,15,17)
connectpoints(X,Y,Z,17,19)
connectpoints(X,Y,Z,15,19)
connectpoints(X,Y,Z,15,21)
connectpoints(X,Y,Z,11,23)
connectpoints(X,Y,Z,23,25)
connectpoints(X,Y,Z,25,27)
connectpoints(X,Y,Z,27,29)
connectpoints(X,Y,Z,29,31)
connectpoints(X,Y,Z,27,31)
connectpoints(X,Y,Z,23,24)
# uncomment below for data labelling
# for x,y,z,i in zip(X,Y,Z,range(len(X))):
#     ax.text(x,y,z,i)
plt.show()

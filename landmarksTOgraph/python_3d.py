from numpy import linspace
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import signal
from matplotlib.animation import FuncAnimation
import sys


TEXTFILE = sys.argv[1]#'lungedata.txt'
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
framedata = np.loadtxt(TEXTFILE, None, None, delimiter=" ")
frame_num = int(len(framedata)/33)
#print(frame_num)
data = np.reshape(framedata, (frame_num, 33, 4))


def connectpoints(X,Y,Z,p1,p2):
    x1, x2 = X[p1], X[p2]
    y1, y2 = Y[p1], Y[p2]
    z1, z2 = Z[p1], Z[p2]
    ax.plot([x1,x2],[y1,y2],[z1,z2],'k-')


def display_points(i):  
    ID = data[i,:,0]
    X = data[i,:,1]
    Y = data[i,:,2]
    Z = data[i,:,3]
    ax.scatter(X, Y, Z, alpha=1)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
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


def animate(i):
    ax.clear()
    display_points(i)


def main():
    ax.view_init(azim=-90, elev=-90)
    ani = FuncAnimation(fig, animate, frame_num,
                        interval=.2, repeat=True)
    plt.show()


if __name__ == "__main__":
    main()
import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Name the video file you want to import for analysis
# Include the file type eg. 'lunges.mp4'
VIDEOFILE = 'squatbackview.mp4'

# Choose a name for the coordinates text file
# Include file type eg. 'lungedata.txt'
TEXTFILE = 'backsquat.txt'

# Choose a name for the frame image files that will create the gif
# Don't add file type eg. 'LungeAnimation'
PICS = 'BackSquat'

# Choose a name for the gif that will be generated from the PICS
GIF = 'BackSquat.gif'

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_connections = mp.solutions.pose.POSE_CONNECTIONS

poselandmarks_list = []
for idx, elt in enumerate(mp_pose.PoseLandmark):
    lm_str = repr(elt).split('.')[1].split(':')[0]
    poselandmarks_list.append(lm_str)

file = VIDEOFILE
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    	# Create VideoCapture object
    cap = cv2.VideoCapture(file)

	# Raise error if file cannot be opened
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        raise TypeError

	# Get the number of frames in the video
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    f=open(TEXTFILE, 'a')

    # For each image in the video, extract the spatial pose data and save it in the appropriate spot in the `data` array 
    frame_num = 0
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        # Create a NumPy array to store the pose data as before
        # The shape is 33x3 - 3D XYZ data for 33 landmarks
        data = np.empty((len(poselandmarks_list), 4))
        np.set_printoptions(suppress=True)

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        landmarks = results.pose_world_landmarks.landmark
        for i in range(len(mp_pose.PoseLandmark)):
            data[i, :] = (i,landmarks[i].x, landmarks[i].y, landmarks[i].z)
        
        # Append array to text file
        np.savetxt(f, data, fmt='%f', newline="\n")
        frame_num += 1
    
    # Close the video file
    cap.release()
    f.close()

framedata = np.loadtxt(TEXTFILE, None, None, delimiter=" ")
data = np.reshape(framedata, (frame_num, 33, 4))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
def connectpoints(X,Y,Z,p1,p2):
    x1, x2 = X[p1], X[p2]
    y1, y2 = Y[p1], Y[p2]
    z1, z2 = Z[p1], Z[p2]
    ax.plot([x1,x2],[y1,y2],[z1,z2],'k-')

# Generate frames 
for i in range(frame_num):
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(azim=-90, elev=-90)
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
    for x,y,z,j in zip(X,Y,Z,range(len(X))):
        ax.text(x,y,z,j)
    plt.savefig(PICS + str(i) + '.png')
    plt.clf()

# Create the gif
with imageio.get_writer(GIF, mode='i', fps=30) as writer:
    for i in range(frame_num):
        image = imageio.imread(PICS + str(i) + '.png')
        writer.append_data(image)

# Deletes frame images if not needed
# Comment out section below if you wish to keep the frame images    
for i in range(frame_num):
    os.remove(PICS + str(i) + '.png')

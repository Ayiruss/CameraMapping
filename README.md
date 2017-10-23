
# CameraMapping
Map the image co-ordinate to the camera and find the camera location and orientation

### INSTRUCTIONS: 
1.  Clone or Download the repository
2. cd CameraMapping
3. run python main.py


### OVERVIEW
Initially I am removing the image pattern for which we know the object location (i.e) QR code as follows



Then I am finding the boxes using contour detection RETR_TREE which has all the hierarchies in the tree structure. 
We know the boxes have exactly five hierarchies, so getting those and find the corners for those contours gives us the vertices of the box.
Same way I have found those locations in pattern.png. 

Solving those object points and corresponding image points using SolvePnP gives the rotation vector and translation vector.


Rotation vector has three degrees of freedom, so we are representing it as vector. Inorder to get the actual angle we need the rotation matrix and for that we are using Rodrigues function. Note: the rotation matrix need not to be unique.


After finding the rotation matrix and translation vector we need to find the camera matrix and we know that

Tranpose(R) * T + C = 0

So, C = - transpose(R) * T

Now we just need to find the pitch, yaw and roll. It can be found using the following
Sr = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
x = math.atan2(R[2,1], R[2,2])
y = math.atan2(-R[2,0], Sr)
z = 0




# RESULT:
### The results will be shown as below

Starting..... IMG_6719.JPG
Displaying the results of : IMG_6719.JPG
Camera - Coordinates :  [[  672.40820312]
 [ 2220.8996582 ]
 [-1390.10229492]]
Pitch Yaw & Roll :  [-1.14122837  0.0781502   0.        ]


Starting..... IMG_6720.JPG
Displaying the results of : IMG_6720.JPG
Camera - Coordinates :  [[ 2111.390625  ]
 [  444.40087891]
 [ 2378.86645508]]
Pitch Yaw & Roll :  [-2.8943341  0.8695884  0.       ]


Starting..... IMG_6721.JPG
Displaying the results of : IMG_6721.JPG
Camera - Coordinates :  [[  271.11004639]
 [ 2320.65161133]
 [-2664.95385742]]
Pitch Yaw & Roll :  [-0.76588933  0.2207567   0.        ]


Starting..... IMG_6722.JPG
Displaying the results of : IMG_6722.JPG
Camera - Coordinates :  [[ -695.76300049]
 [ 1350.68127441]
 [-2663.98535156]]
Pitch Yaw & Roll :  [-0.32211317 -0.12278705  0.        ]
Starting..... IMG_6723.JPG
Displaying the results of : IMG_6723.JPG
Camera - Coordinates :  [[ 2224.2746582 ]
 [  704.01721191]
 [ 2529.11181641]]
Pitch Yaw & Roll :  [-2.86440815  0.48497686  0.        ]
Starting..... IMG_6724.JPG

Displaying the results of : IMG_6724.JPG
Camera - Coordinates :  [[ 2196.09472656]
 [  420.73568726]
 [ 2516.07421875]]
Pitch Yaw & Roll :  [-3.11305928  0.49282649  0.        ]


Starting..... IMG_6725.JPG
Displaying the results of : IMG_6725.JPG
Camera - Coordinates :  [[ 2104.94311523]
 [  380.18029785]
 [ 2519.33349609]]
Pitch Yaw & Roll :  [-3.09831458  0.6815719   0.        ]


Starting..... IMG_6726.JPG
Displaying the results of : IMG_6726.JPG
Camera - Coordinates :  [[ 2183.15820312]
 [  437.28659058]
 [ 2526.17724609]]
Pitch Yaw & Roll :  [-3.13662476  0.5032108   0.        ]


Starting..... IMG_6727.JPG
Displaying the results of : IMG_6727.JPG
Camera - Coordinates :  [[-1112.07141113]
 [ 1038.65795898]
 [-2747.015625  ]]
Pitch Yaw & Roll :  [-0.59734488 -0.42191205  0.        ]



# CameraMapping
Map the image co-ordinate to the camera and find the camera location and orientation
# CameraMapping
Map the image co-ordinate to the camera and find the camera location and orientation

###INSTRUCTIONS: 
1.  Clone or Download the repository
2. cd CameraMapping
3. run python main.py


###OVERVIEW
Initially I am removing the image pattern for which we know the object location (i.e) QR code as follows

Then I am finding the boxes using contour detection RETR_TREE which has all the hierarchies in the tree structure. We know the boxes have exactly five hierarchies, so getting those and find the corners for those contours gives us the vertices of the box.
Same way I have found those locations in pattern.png. Solving those object points and corresponding image points using SolvePnP gives the rotation vector and translation vector.
Rotation vector has three degrees of freedom, so we are representing it as vector. Inorder to get the actual angle we need the rotation matrix and for that we are using Rodrigues function. Note: the rotation matrix need not to be unique.
After finding the rotation matrix and translation vector we need to find the camera matrix and we know that
Tranpose(R) * T + C = 0
So, C = - transpose(R) * T
Now we just need to find the pitch, yaw and roll. It can be found using the following
Sr = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
x = math.atan2(R[2,1], R[2,2])
y = math.atan2(-R[2,0], Sr)
z = 0















###RESULT:
The results will be shown as below

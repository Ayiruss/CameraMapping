import cv2
import numpy as np
import math
import os

#To Extract the QR part of the image for further processing.
def getQRFromImage(image):

	imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh, bw = cv2.threshold(imgray, 240,255,0)
	edge = cv2.Canny(bw, thresh/2, thresh)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
	morph = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
	total = 0
	_, contours, hierarchy = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for cnt in contours:
		peri = cv2.arcLength(cnt,True)
		approx = cv2.approxPolyDP(cnt,0.02*peri, True)

		if len(approx) ==  4:
			cv2.drawContours(image, [approx], -1, (0,255,0), 4)
			total +=1
		x,y,w,h = cv2.boundingRect(cnt)
		QR = image[y:y+h,x:x+w]

	return x,y,QR

def cross(p1, p2):
	return (p1[0]*p2[1] - p1[1]*p2[0])

#Intersection to find one of the corner wihtout the box
def getIntersection(v1, v2, v3, v4):
	p = v1
	q = v3
	r = v2 - v1
	s = v4 - v3
	if cross(r,s) == 0:
		return False
	t = cross(q-p,s)/cross(r,s)
	intersection = p + t*r
	return True, intersection

#To find the corner from the contour by finding maximum distance from center
def getCorner(P, ref, baseline, corner):
	distance = getDistance(P[0],ref)
	if distance > baseline:
		baseline = distance
		corner = P[0]
	return baseline, corner

#returns the four corner vertices of the given contour.
def getVertices(contours, c_id, slope):
	vertices = []
	x, y, w, h = cv2.boundingRect(contours[c_id])

	L = []
	K = []
	M = []
	N = []
	W = []
	X = []
	Y = []
	Z = []
	P0 = []
	P1 = []
	P2 = []
	P3 = []

	maxDistance = np.zeros(4, np.float32)

	K = (x,y)
	L.append(x+w)
	L.append(y)
	M = ((x+w, y+h))
	N.append(x)
	N.append(y+h)

	W.append((K[0] + L[0]) / 2)
	W.append(K[1])

	X.append(L[0])
	X.append((L[1] + M[1]) / 2)

	Y.append((M[0] + N[1]) / 2)
	Y.append(M[1])

	Z.append(N[0])
	Z.append((N[1] + K[1]) / 2)

	dist1 = 0.0
	dist2 = 0.0
	
	if (int(slope) > 5) or (int(slope) < -5):
		for i in range(len(contours[c_id])):
			dist1 = getLineDistance(M,K,contours[c_id][i][0])
			dist2 = getLineDistance(L,N,contours[c_id][i][0])
			if (dist1 >= 0.0) and (dist2 >= 0.0):
				maxDistance[1], P1 = getCorner(contours[c_id][i], W, maxDistance[1], P1)
			elif (dist1 > 0.0) and (dist2 <= 0.0):
				maxDistance[2], P2 = getCorner(contours[c_id][i], X, maxDistance[2], P2)	
			elif (dist1 <= 0.0) and (dist2 < 0.0):
				maxDistance[3], P3 = getCorner(contours[c_id][i], Y, maxDistance[3], P3)		
			elif (dist1 < 0.0) and (dist2 >= 0.0):
				maxDistance[0], P0 = getCorner(contours[c_id][i], Z, maxDistance[0], P0)
			else:
				continue
	else:
		halfx = (L[0] + K[0]) / 2
		halfy = (L[1] + N[1]) / 2
		for i in range(len(contours[c_id])):
			if (contours[c_id][i][0][0] < int(halfx)) and (contours[c_id][i][0][1] <= int(halfy)):
				maxDistance[2], P0 = getCorner(contours[c_id][i], M, maxDistance[2], P0)
			elif (contours[c_id][i][0][0] >= int(halfx)) and (contours[c_id][i][0][1] < int(halfy)):
				maxDistance[3], P1 = getCorner(contours[c_id][i], N, maxDistance[3], P1)
			elif (contours[c_id][i][0][0] > int(halfx)) and (contours[c_id][i][0][1] >= int(halfy)):
				maxDistance[0], P2 = getCorner(contours[c_id][i], K, maxDistance[0], P2)
			elif (contours[c_id][i][0][0] <= int(halfx)) and (contours[c_id][i][0][1] > int(halfy)):
				maxDistance[1], P3 = getCorner(contours[c_id][i], L, maxDistance[1], P3)

	vertices.append(P0)
	vertices.append(P1)
	vertices.append(P2)
	vertices.append(P3)
	return vertices

#Updates the corner based on the orientation of the image
def updateCorner(orientation, IN):
	OUT = []
	P0 = []
	P1 = []
	P2 = []
	P3 = []

	if orientation == QR_TOP:
		P0 = IN[0]
		P1 = IN[1]
		P2 = IN[2]
		P3 = IN[3]
	elif orientation == QR_RIGHT:
		P0 = IN[1]
		P1 = IN[2]
		P2 = IN[3]
		P3 = IN[0]
	elif orientation == QR_BOTTOM:
		P0 = IN[2]
		P1 = IN[3]
		P2 = IN[0]
		P3 = IN[1]
	elif orientation == QR_LEFT:
		P0 = IN[3]
		P1 = IN[0]
		P2 = IN[1]
		P3 = IN[2]

	OUT.append(P0)
	OUT.append(P1)
	OUT.append(P2)
	OUT.append(P3)
	return OUT

#returns slope between two points
def getLineSlope(P1, P2):
	dx = P2[0] - P1[0]
	dy = P2[1] - P1[1]
	alignment = 0
	if int(dy) != 0:
		alignment =1 
		return alignment, (dy/dx)
	return alignment, 0.0

#Perpendicular Distance of the point P3 from line formed by P1 and P2
def getLineDistance(P1, P2, P3):
	a = -((P2[1] - P1[1]) / (P2[0] - P1[0]))
	b = 1.0
	c = (((P2[1] - P1[1]) / (P2[0] - P1[0])) * P1[0]) - P1[1]
	distance = (a * P3[0] + (b * P3[1]) + c) / math.sqrt((a*a) + (b*b))
	return distance

#returns the distance between two points
def getDistance(p1, p2):
	return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

#verifies Rotation matrix by checking Rt.R = I
def isRotationMatrix(R):
	Rt = np.transpose(R)
	dotRtR = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - dotRtR)

	return n < 1e-6

#Converts the rotation matrix to Pitch,  Yaw and roll representation
def rotationMatrixToEulerAngles(R):
	assert(isRotationMatrix(R))

	Sr = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
	singular = Sr < 1e-6

	if not singular:
		x = math.atan2(R[2,1], R[2,2])
		y = math.atan2(-R[2,0], Sr)
		z = 0
	
	return np.array([x,y,z])




#Image Orientations
QR_TOP = 0
QR_RIGHT = 1
QR_BOTTOM = 2
QR_LEFT = 3

#iPhone6 Camera Intrinsics Parameters

f = 4.15
resX = 3264
resY = 2448
sensorSizeX = 4.89
sensorSizeY = 3.67
fx = f * resX/sensorSizeX
fy = f * resY/sensorSizeY
cx = resX/2
cy = resY/2

#holding each squares index

files = [f for f in os.listdir('.') if os.path.isfile(f) and f[-3:] == 'JPG']

for f in files:
	print("Starting.....", f)
	image = cv2.imread(f)
	A = 0
	B = 0
	C = 0
	top = 0
	middle_box = 0
	left_box = 0 
	right_box = 0
	orientation = 0
	QR_x, QR_y, QR = getQRFromImage(image)
	gray = cv2.cvtColor(QR, cv2.COLOR_BGR2GRAY)
	edge = cv2.Canny(gray, 100, 200)

	edge, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	DBG = 1
	total = 0
	centers = []
	level = 0
	for cnt in contours:
		moment = cv2.moments(cnt)
		c_x = 0
		c_y = 0
		if int(moment['m00']) != 0:
			c_x = (int(moment['m10']/moment['m00']))
			c_y = (int(moment['m01']/moment['m00']))
		centers.append((c_x,c_y))

	i = 0
	for cnt in contours:
		box = cv2.boundingRect(cnt)
		count = 0
		k = i
		epsilon = 0.02 * cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, epsilon, True)

		if len(approx) == 4:
			total = total + 1
			while hierarchy[0][k][2] != -1:
				k = hierarchy[0][k][2]
				count = count + 1

			if hierarchy[0][k][2] != -1:
				count = count + 1

			if count == 5:
				if level == 0:
					A = i
				elif level ==1:
					B = i
				elif level == 2:
					C = i

				level = level + 1
		i = i + 1


	if level >= 3:
		AB = getDistance(centers[A], centers[B])
		BC = getDistance(centers[B], centers[C])
		CA = getDistance(centers[C], centers[A])
		if (int(AB) > int(BC)) and (int(AB) > int(CA)):
			middle_box = C
			left_box = A
			right_box = B
		elif (int(CA) > int(AB)) and (int(CA) > int(BC)):
			middle_box = B
			left_box = A
			right_box = C
		elif (int(BC) > int(AB)) and (int(BC) > int(CA)):
			middle_box = A
			left_box = B
			right_box = C

		top = middle_box

		distance = getLineDistance(centers[left_box], centers[right_box], centers[middle_box])
		align, slope = getLineSlope(centers[left_box], centers[right_box])

		if align == 0:
			bottom = left_box
			rigth = right_box
		elif (int(slope) <= 0) and (int(distance) < 0):
			bottom = left_box
			right = right_box
			orientation = QR_TOP
		elif (int(slope) >= 0) and (int(distance) < 0):
			right = left_box
			bottom = right_box
			orientation = QR_RIGHT
		elif (int(slope) <= 0) and (int(distance) > 0):
			right = left_box
			bottom = right_box
			orientation = QR_BOTTOM
		elif (int(slope) >= 0) and (int(distance) > 0):
			bottom = left_box
			right = right_box
			orientation = QR_LEFT

		if (top < len(contours)) and (right < len(contours)) and (bottom < len(contours)):
			#Corners
			C1 = []
			C2 = []
			C3 = []
			C4 = []

			c1 = []
			c2 = []
			c3 = []
			
			#vertices of each boxes in the QR
			c1 = getVertices(contours, top, slope)
			c2 = getVertices(contours, right, slope)
			c3 = getVertices(contours, bottom, slope)

			

			C1 = updateCorner(orientation, c1)
			C2 = updateCorner(orientation, c2)
			C3 = updateCorner(orientation, c3)
		
			iflag, N = getIntersection(C2[1], C2[2], C3[3], C3[2])

			
	#The object points are found by running the same code using pattern.jpg
	objectPoints = []
	objectPoints.append([41,39,0])
	objectPoints.append([108,39,0])
	objectPoints.append([39,108,0])
	objectPoints.append([221,39,0])
	objectPoints.append([221,39,0])
	objectPoints.append([288,39,0])
	objectPoints.append([289,109,0])
	objectPoints.append([219,108,0])
	objectPoints.append([41,219,0])
	objectPoints.append([108,219,0])
	objectPoints.append([109,289,0])
	objectPoints.append([39,288,0])
	objectPoints.append([291.6,291.6,0])
	objectPoints = np.array(objectPoints, np.float32)

	#The points in C1,C2,C3 has the each corner of the boxes and C4 has the independent corner adding QR_x and QR_y get the actual position of the boxes in the image
	imagePoints = []
	for point in C1:
		imagePoints.append((point[0] + QR_x, point[1] + QR_y))
	for point in C2:
		imagePoints.append((point[0] + QR_x, point[1] + QR_y))
	for point in C3:
		imagePoints.append((point[0] + QR_x, point[1] + QR_y))
	imagePoints.append((N[0] + QR_x, N[1] + QR_y))
	imagePoints = np.array(imagePoints, np.float32)


	#Camera matrix of iPhone 6 which is calculated manually.
	camera_matrix = np.zeros((3,3), np.float32)
	camera_matrix[0][0] = fx
	camera_matrix[1][1] = fy
	camera_matrix[0][2] = cx
	camera_matrix[1][2] = cy
	camera_matrix[2][2] = 1

	#rvec, tvec = cv2.solvePnP(np.array(objectPoints).astype('float32'), np.array(imagePoints).astype('float32'),np.array(camera_matrix).astype('float32'),None)[-2:]

	#rotation vector and translation vector
	rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, camera_matrix, None)[-2:]

	#rotation matrix corresponding to the rotatoin vector.
	rotation = cv2.Rodrigues(rvec)[0]

	#
	rotation_tr = np.transpose(rotation)


	#RC + T = 0 -> C = -R-1 * T
	camera_coordinates = np.matmul(-rotation_tr,tvec)

	#Pitch, Yaw and Roll of the camera
	angle = rotationMatrixToEulerAngles(rotation)
	print('---------------------------------------------------------------------------------------------------------')
	print("Displaying the results of : " + f)
	print("Camera - Coordinates : ", camera_coordinates)
	print("Pitch Yaw & Roll : ", angle )

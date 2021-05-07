from djitellopy import tello
from time import sleep
import numpy as np
import math
import cv2
import keyPressModule as kp

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

fSpeed = 117 / 10  # Forward Speed in cm/s   (15cm/s)

aSpeed = 360 / 10  # Angular Speed Degrees/s  (50d/s)

interval = 0.25

dInterval = fSpeed * interval

aInterval = aSpeed * interval

x, y = 500, 500

a = 0

yaw = 0

points = []

def getKeyboardInput():
	leftRight, fowardBack, upDown, yawVelocity = 0,0,0,0
    
	speed = 70

	aspeed = 50

	d = 0

	global x, y, yaw, a

	if kp.getKey("LEFT"): 
		leftRight = -speed
		d = dInterval
		a = -180
	elif kp.getKey("RIGHT"): 
		leftRight = speed
		d = -dInterval
		a = 180
	
	if kp.getKey("UP"): 
		fowardBack = speed
		d = dInterval
		a = 270
	
	elif kp.getKey("DOWN"): 
		fowardBack = -speed
		d = -dInterval
		a = -90
	
	if kp.getKey("w"): upDown = speed
	elif kp.getKey("s"): upDown = -speed
	
	if kp.getKey("a"): 
		yawVelocity = -speed
		yaw += aInterval
	elif kp.getKey("d"): 
		yawVelocity = speed
		yaw -= aInterval
	
	if kp.getKey("q"): me.land(); sleep(3)
	if kp.getKey("e"): me.takeoff()

	sleep(interval)
	a += yaw
	x += int(d*math.cos(math.radians(a)))
	y += int(d*math.sin(math.radians(a)))
	return [leftRight, fowardBack, upDown, yawVelocity, x, y]

def drawPoints(imgMap, points):
	for point in points:
		cv2.circle(imgMap,point,10,(255,0,0),cv2.FILLED)

while True:
	img = me.get_frame_read().frame
	img = cv2.resize(img,(1440,960))
	cv2.imshow("Image", img)
	vals = getKeyboardInput()
	me.send_rc_control(vals[0], vals[1], vals[2], vals[3])

	# imgMap = np.zeros((1000,1000,3), np.uint8)
	# points.append((vals[4], vals[5]))
	# drawPoints(imgMap, points)
	# cv2.imshow('Output', imgMap)
	# cv2.waitkey(1)

	
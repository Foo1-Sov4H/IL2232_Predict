import math
import numpy as np
import time

def PredictVelocity(v1,v2,v3):
	#v1,v2: historical data
	#v3: current data

	start_time = time.time()

	diff_v1 = v3 - v2
	diff_v2 = v2 - v1
	diff_acc = diff_v2 - diff_v1 
	predict_v = diff_acc + diff_v2 + v3

	end_time = time.time()
	execution_time = end_time - start_time

	return predict_v , execution_time

def Judgement(x1 , y1 , x2 , y2 , x3 , y3):

	slope1 = (y2 - y1) / (x2 - x1)
	slope2 = (y3 - y2) / (x3 - x2)
	if (x1==x2 and y1==y2):
		slope1 = float('inf')
	if (x2==x3 and y2==y3):
		slope2 = float('inf')

	h1 = math.atan(slope1)
	h2 = math.atan(slope2)
	
	if (h1 - h2) ** 2 < 3:
		return 0	
	else:
		return 1
	

def PredictLocationLine(x1 , y1 , x2 , y2 , x3 , y3 , v3 , predict_v):
	#(x1,y1) (x2,y2): historical data
	#(x3,y3): current data

	start_time = time.time()

	d_x = 0.1 * 0.5 * (predict_v + v3)
	slope1 = (y2 - y1) / (x2 - x1)
	slope2 = (y3 - y2) / (x3 - x2)
	if (x1==x2 and y1==y2):
		slope1 = float('inf')
	if (x2==x3 and y2==y3):
		slope2 = float('inf')
	
	d_e = math.sqrt(d_x ** 2 / (1 + slope2 ** 2))
	d_n = slope1 * d_e
	predict_x = d_e + x3
	predict_y = d_n + y3
	if ((predict_x - x2) ** 2 + (predict_y - y2) ** 2) < d_x * d_x :
		predict_x = x3 - d_e 
		predict_y = y3 - d_n

	end_time = time.time()

	execution_time = end_time - start_time
	return predict_x , predict_y, execution_time

def PredictLocationCircle(x1 , y1 , x2 , y2 , x3 , y3 , v3 , predict_v):
	#(x1,y1) (x2,y2): historical data
	#(x3,y3): current data
	start_time = time.time()

	d_x = 0.1 * 0.5 * (predict_v + v3)
	slope1 = (y2 - y1) / (x2 - x1)
	slope2 = (y3 - y2) / (x3 - x2)

	midpoint1 = ((x1 + x2) / 2, (y1 + y2) / 2)
	midpoint2 = ((x2 + x3) / 2, (y2 + y3) / 2)
	if (x1==x2 and y1==y2):
		slope1 = float('inf')
	if (x2==x3 and y2==y3):
		slope2 = float('inf')

	center_x = (midpoint2[1] - midpoint1[1] + slope1 * midpoint1[0] - slope2 * midpoint2[0]) / (slope1 - slope2)
	center_y = slope1 * (center_x - midpoint1[0]) + midpoint1[1]

	radius = math.sqrt((center_x - x1) ** 2 + (center_y - y1) ** 2)

	theta = math.atan2(y3 - center_y, x3 - center_x)
	
	theta_new = theta + d_x / radius

	predict_x = center_x + radius * math.cos(theta_new)
	predict_y = center_y + radius * math.sin(theta_new)

	if  ((predict_x - x2) ** 2 + (predict_y - y2) ** 2) < d_x * d_x :
		theta_new = theta - d_x / radius
		predict_x = center_x + radius * math.cos(theta_new)
		predict_y = center_y + radius * math.sin(theta_new)

	end_time = time.time()

	execution_time = end_time - start_time
	return predict_x , predict_y, execution_time


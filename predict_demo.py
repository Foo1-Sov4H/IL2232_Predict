import math
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# get TV data
df = pd.read_csv('JRC-VC_260219_part2.csv')
features = ['Time', 'Speed1', 'E1', 'N1']
data = df[features]
predict_v = []
real_v = []
time_list = []
real_e = []
real_n = []
predict_e = []
predict_n = []
inaccuracies = []

#print(data.iloc[1:20])

for i in range(1000):

	#Predict Speed

	time_list.append(0.2+0.1*i)
	start_time_v = time.time()

	diff_speed1 = data.iloc[i+3,1] - data.iloc[i+2,1]
	diff_speed2 = data.iloc[i+2,1] - data.iloc[i+1,1]
	diff_speed = 2 * diff_speed1 - diff_speed2 
	predict_speed = diff_speed + data.iloc[i+3,1]
	predict_v.append(predict_speed)

	end_time_v = time.time()
	execution_time = end_time_v - start_time_v

	print("Predicted Time of Speed:")
	print(execution_time)

	real_v.append(data.iloc[i+4,1])

	#Predict Location
	point1 = (data.iloc[i+1,2], data.iloc[i+1,3])
	point2 = (data.iloc[i+2,2], data.iloc[i+2,3])
	point3 = (data.iloc[i+3,2], data.iloc[i+3,3])
	point4 = (data.iloc[i+4,2], data.iloc[i+4,3])
	
	real_e.append(data.iloc[i+4,2])
	real_n.append(data.iloc[i+4,3])

	start_time_l = time.time()
	d_x = 0.1 * 0.5 * (predict_speed + data.iloc[i+3,1])

	slope1 = (point2[1] - point1[1]) / (point2[0] - point1[0])
	slope2 = (point3[1] - point2[1]) / (point3[0] - point2[0])
	#print(point1)
	#print(point2)
	#print(point3)
	#print(point4)

	d_e = math.sqrt(d_x ** 2 / (1 + slope2 ** 2))
	d_n = slope1 * d_e
	predict_e = d_e + data.iloc[i+3,2]
	predict_n = d_n + data.iloc[i+3,3]
	if ((predict_e - point2[0]) ** 2 + (predict_n - point2[1]) ** 2) < d_x * d_x :
		predict_e = data.iloc[i+3,2] - d_e 
		predict_n = data.iloc[i+3,3] - d_n

	#print("Inaccuracies:")
	#inaccuracies = math.sqrt((predict_e - data.iloc[i+4,2]) ** 2 + (predict_n - data.iloc[i+4,3]) ** 2)
	#point_pre = (predict_e, predict_n)
	#print(inaccuracies)
	#print("predict location:")
	#print(point_pre)
	


	if (math.atan(slope2) - math.atan(slope1)) ** 2 < 5:
		d_e = math.sqrt(d_x ** 2 / (1 + slope1 ** 2))
		d_n = slope1 * d_e
		predict_e = d_e + data.iloc[i+3,2]
		predict_n = d_n + data.iloc[i+3,3]
		if ((predict_e - point2[0]) ** 2 + (predict_n - point2[1]) ** 2) < d_x * d_x :
			predict_e = data.iloc[i+3,2] - d_e 
			predict_n = data.iloc[i+3,3] - d_n
	else:
		midpoint1 = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)
		midpoint2 = ((point2[0] + point3[0]) / 2, (point2[1] + point3[1]) / 2)
		if point1[0] == point2[0]:
			slope1 = float('inf')
		if point2[0] == point3[0]:
			slope2 = float('inf')

		center_x = (midpoint2[1] - midpoint1[1] + slope1 * midpoint1[0] - slope2 * midpoint2[0]) / (slope1 - slope2)
		center_y = slope1 * (center_x - midpoint1[0]) + midpoint1[1]

		radius = math.sqrt((center_x - point1[0]) ** 2 + (center_y - point1[1]) ** 2)

		theta = math.atan2(point3[1] - center_y, point3[0] - center_x)
		
		theta_new = theta + d_x / radius

		predict_e = center_x + radius * math.cos(theta_new)
		predict_n = center_y + radius * math.sin(theta_new)

		if  ((predict_e - point2[0]) ** 2 + (predict_n - point2[1]) ** 2) < d_x * d_x :
			theta_new = theta - d_x / radius
			predict_e = center_x + radius * math.cos(theta_new)
			predict_e = center_y + radius * math.sin(theta_new)
	
	end_time_l = time.time()
	execution_time_l = end_time_l - start_time_l

	print("Predicted Time of Location:")
	print(execution_time_l)

	inaccuracie = math.sqrt((predict_e - data.iloc[i+4,2]) ** 2 + (predict_n - data.iloc[i+4,3]) ** 2)
	inaccuracies.append(inaccuracie)

	#point_pre = (predict_e, predict_n)
	#print("predict location:")
	#print(point_pre)	
	#print(predict_speed)

#plt.plot(time_list, predict_v, color='red')
#plt.plot(time_list, real_v, color='blue')
plt.plot(time_list, inaccuracies, color='blue')
plt.xlabel('Time')
#plt.ylabel('Speed')
plt.ylabel('Inaccuracie')
plt.show()

	

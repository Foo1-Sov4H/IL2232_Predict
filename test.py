import math
import pandas as pd
import time
import actr_predict as actr
import matplotlib.pyplot as plt

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
	time_list.append(0.2+0.1*i)

	#Predict Speed

	v1 = data.iloc[i+1,1]
	v2 = data.iloc[i+2,1]
	v3 = data.iloc[i+3,1]
	(predict_speed , execution_time) = actr.PredictVelocity(v1,v2,v3)
	predict_v.append(predict_speed)

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
	if point1[0] == point2[0]:
		slope1 = float('inf')
	if point2[0] == point3[0]:
		slope2 = float('inf')

	h1 = math.atan(slope1)
	h2 = math.atan(slope2)


	if (h1 - h2) ** 2 < 3:
		(predict_e , predict_n , execution_time) = actr.PredictLocationLine(data.iloc[i+1,2], data.iloc[i+1,3], data.iloc[i+2,2], data.iloc[i+2,3], data.iloc[i+3,2], data.iloc[i+3,3], v3, predict_speed)
	else:
		(predict_e , predict_n , execution_time) = actr.PredictLocationCircle(data.iloc[i+1,2], data.iloc[i+1,3], data.iloc[i+2,2], data.iloc[i+2,3], data.iloc[i+3,2], data.iloc[i+3,3], v3, predict_speed)


	print("Predicted Time of Location:")
	print(execution_time)

	inaccuracy = math.sqrt((predict_e - data.iloc[i+4,2]) ** 2 + (predict_n - data.iloc[i+4,3]) ** 2)
	inaccuracies.append(inaccuracy)


plt.plot(time_list, predict_v, color='red')
plt.plot(time_list, real_v, color='blue')
#plt.plot(time_list, inaccuracies, color='blue')
plt.xlabel('Time')
plt.ylabel('Speed')
#plt.ylabel('Inaccuracie')
plt.show()

	

import math
import numpy as np
import pandas as pd
import time
from hmmlearn import hmm

def distance2line(road_width, road_position, car_width, car_length, car_position, car_velocity):

	# output :: distance2line_left, distance2line_right, result_left, result_right
	## distance2line_left :: The shortest distance between the target vehicle and the left side of the lane
	## distance2line_right :: The shortest distance between the target vehicle and the right side of the lane
	## result_left :: 1 - The top of car is closer to the left side of the lane
			## 2 - The bot of car is closer to the left side of the lane
	## result_right :: 1 - The top of car is closer to the right side of the lane
			## 2 - The bot of car is closer to the right side of the lane

	direction_vector = [car_velocity[0], car_velocity[1]]
	half_length = car_length / 2
	half_width = car_width / 2
	position_top_left = car_position + ( half_length * direction_vector[0] - half_width * direction_vector[1] , 
					half_length * direction_vector[1] + half_width * direction_vector[0])
	position_top_right = car_position + ( half_length * direction_vector[0] + half_width * direction_vector[1],
                    half_length * direction_vector[1] - half_width * direction_vector[0])
	position_bot_left = car_position + ( - half_length * direction_vector[0] - half_width * direction_vector[1],
                    - half_length * direction_vector[1] + half_width * direction_vector[0])
	position_bot_right = car_position + ( - half_length * direction_vector[0] + half_width * direction_vector[1],
                    - half_length * direction_vector[1] - half_width * direction_vector[0])
	
	road_position_left = road_position - road_width / 2
	road_position_right = road_position + road_width / 2

	distance2line_left = min(position_top_left[1] - road_position_left , position_bot_left[1] - road_position_left)
	distance2line_right = min( road_position_right - position_top_right[1] , road_position_right - position_bot_right[1])

	if (distance2line_left == position_top_left[1] - road_position_left):
		result_left = 1
	else:
		result_left = 2

	if (distance2line_right == road_position_right - position_top_right[1]):
		result_right = 1
	else:
		result_right = 2

	return distance2line_left, distance2line_right, result_left, result_right



def ChangeStartPoint(road_width, road_position, car_width, car_length, car_velocity_0, car_position_0, car_velocity_1, car_position_1):

	(distance2line_left_0, distance2line_right_0, result_left_0, result_right_0) =	\
		distance2line(road_width, road_position, car_width, car_length, car_position_0, car_velocity_0)
	(distance2line_left_1, distance2line_right_1, result_left_1, result_right_1) =	\
		distance2line(road_width, road_position, car_width, car_length, car_position_1, car_velocity_1)
	
	if (distance2line_right_0 > 0) and (distance2line_right_1 <= 0) \
		and (result_right_0 == 1) and (result_right_1 == 1):
		return 1	## change to right lane at this time point

	if (distance2line_left_0 > 0) and (distance2line_left_1 <= 0) \
		and (result_left_0 == 1) and (result_left_1 == 1):
		return 2	## change to left lane at this time point

	return 0 ## dont change lane at this time point


def ChangeCheck(road_width, road_position, car_width, car_length, car_position, car_velocity):
	direction_vector = [car_velocity[0], car_velocity[1]]
	half_length = car_length / 2
	half_width = car_width / 2
	position_top_left = car_position + ( half_length * direction_vector[0] - half_width * direction_vector[1] , 
					half_length * direction_vector[1] + half_width * direction_vector[0])
	position_top_right = car_position + ( half_length * direction_vector[0] + half_width * direction_vector[1],
                    half_length * direction_vector[1] - half_width * direction_vector[0])
	position_bot_left = car_position + ( - half_length * direction_vector[0] - half_width * direction_vector[1],
                    - half_length * direction_vector[1] + half_width * direction_vector[0])
	position_bot_right = car_position + ( - half_length * direction_vector[0] + half_width * direction_vector[1],
                    - half_length * direction_vector[1] - half_width * direction_vector[0])
	
	road_position_left = road_position - road_width / 2
	road_position_right = road_position + road_width / 2

	if (position_top_left[1] > road_position_left) and (position_top_left[1] < road_position_right) and \
		(position_top_right[1] > road_position_left) and (position_top_right[1] < road_position_right) and \
		(position_bot_left[1] > road_position_left) and (position_bot_left[1] < road_position_right) and \
		(position_bot_right[1] > road_position_left) and (position_bot_right[1] < road_position_right):
		return 0 ## not changing lane this time
	else:
		return 1 ## TV is changing lane this time


def ChangeEndPoint(road_width, road_position, car_width, car_length, car_velocity_0, car_position_0, car_velocity_1, car_position_1):
	
	(distance2line_left_0, distance2line_right_0, result_left_0, result_right_0) =	\
		distance2line(road_width, road_position, car_width, car_length, car_position_0, car_velocity_0)
	(distance2line_left_1, distance2line_right_1, result_left_1, result_right_1) =	\
		distance2line(road_width, road_position, car_width, car_length, car_position_1, car_velocity_1)
	
	if (distance2line_right_0 < 0) and (distance2line_right_1 >= 0) \
		and (result_right_0 == 2) and (result_right_1 == 2):
		return 1	## finish changing to left lane at this time point

	if (distance2line_left_0 < 0) and (distance2line_left_1 >= 0) \
		and (result_left_0 == 2) and (result_left_1 == 2):
		return 2	## finish changing to right lane at this time point

	return 0 ## dont finish change lane at this time point


def PredictVelocity(v1,v2,v3):
	#v1,v2: historical data
	#v3: current data

	diff_v1 = v3 - v2
	diff_v2 = v2 - v1
	diff_acc = diff_v2 - diff_v1 
	predict_v = diff_acc + diff_v2 + v3

	return predict_v 

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


	d_x = 0.04 * 0.5 * (predict_v + v3) ##For HighD dataset, 0.04 second per frame
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

	return predict_x , predict_y

def PredictLocationCircle(x1 , y1 , x2 , y2 , x3 , y3 , v3 , predict_v):
	#(x1,y1) (x2,y2): historical data
	#(x3,y3): current data
	start_time = time.time()

	d_x = 0.04 * 0.5 * (predict_v + v3)
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

def HMMModelTrainCenter():
	# Model will trained by velocity_x , velocity_y , and current lane 
	## lane3 : ->
	## lane7 : <-
	df = pd.read_csv('highD-dataset-v1.0/data/59_tracks.csv')

	grouped_data = df.groupby(['id', df.groupby('id')['laneId'].transform('first')])

	data_dict = {}

	for group_name, group_data in grouped_data:
		data_dict[group_name] = group_data

	selected_groups_4 = grouped_data.filter(lambda x: x['laneId'].iloc[0] == 4)
	selected_groups_8 = grouped_data.filter(lambda x: x['laneId'].iloc[0] == 8)

	train_data_4 = selected_groups_4[['x', 'y', 'xVelocity', 'yVelocity', 'laneId']].values
	train_data_8 = selected_groups_8[['x', 'y', 'xVelocity', 'yVelocity', 'laneId']].values

	model4 = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=10000)
	model8 = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=10000)

	model4.fit(train_data_4)
	model8.fit(train_data_8)

	return model4, model8


def HMMModelTrainLeftRight():
	# Model will trained by velocity_x , velocity_y , and current lane 
	## lane2 : right lane & <-
	## lane3 : left  lane & <-
	## lane5 : left  lane & ->
	## lane6 : right lane & ->
	df = pd.read_csv('highD-dataset-v1.0/data/01_tracks.csv')

	grouped_data = df.groupby(['id', df.groupby('id')['laneId'].transform('first')])

	data_dict = {}

	for group_name, group_data in grouped_data:
		data_dict[group_name] = group_data
	
	selected_groups_2 = grouped_data.filter(lambda x: x['laneId'].iloc[0] == 2)
	selected_groups_3 = grouped_data.filter(lambda x: x['laneId'].iloc[0] == 3)
	selected_groups_5 = grouped_data.filter(lambda x: x['laneId'].iloc[0] == 5)
	selected_groups_6 = grouped_data.filter(lambda x: x['laneId'].iloc[0] == 6)


	train_data_2 = selected_groups_2[['x', 'y', 'xVelocity', 'yVelocity', 'laneId']].values
	train_data_3 = selected_groups_3[['x', 'y', 'xVelocity', 'yVelocity', 'laneId']].values
	train_data_5 = selected_groups_5[['x', 'y', 'xVelocity', 'yVelocity', 'laneId']].values
	train_data_6 = selected_groups_6[['x', 'y', 'xVelocity', 'yVelocity', 'laneId']].values


	model2 = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=10000)
	model3 = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=10000)
	model5 = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=10000)
	model6 = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=10000)

	model2.fit(train_data_2)
	model3.fit(train_data_3)
	model5.fit(train_data_5)
	model6.fit(train_data_6)

	return model2, model3, model5, model6


	

def probably2otherlane_c(road_width, road_position, car_width, car_length, car_velocity, car_position):
	# Check whether TV will change its lane in 20 x 0.04 = 0.8 second
	### car_velocity = [(vx0,vy0),(vx1,vy1),(vx2,vy2)]
	### car_position = [(x0,y0),(x1,y1),(x2,y2)]
	trajectory = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
           (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
           (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
           (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),]
	for i in (0,19):
		if ChangeCheck(road_width, road_position, car_width, car_length, car_position, car_velocity[2]) == 1:
			return 0, i ### target vehicle is changing lane now!
		else:
			car_velocity_predict_x = PredictVelocity(car_velocity[0][0],car_velocity[1][0],car_velocity[2][0])
			car_velocity_predict_y = PredictVelocity(car_velocity[0][1],car_velocity[1][1],car_velocity[2][1])
			car_velocity_predict = (car_velocity_predict_x, car_velocity_predict_y)

			car_position_predict_x  = 0.04 * 0.5 * (car_velocity_predict_x + car_velocity[2][0]) + car_position[2][0]
			car_position_predict_y  = 0.04 * 0.5 * (car_velocity_predict_y + car_velocity[2][1]) + car_position[2][1]
			car_position_predict = (car_position_predict_x, car_position_predict_y)

			result_change = ChangeStartPoint(road_width, road_position, car_width, car_length, car_velocity[2], car_position[2], car_velocity_predict, car_position_predict)

			if result_change == 1:
				return 1, i, trajectory  ## target vehicle change to right lane at this time point i
			if result_change == 2:
				return 2, i, trajectory   ## target vehicle change to left lane at this time point i
			if result_change == 0:
				trajectory[i] = car_position_predict
				car_velocity[0] = car_velocity[1]
				car_velocity[1] = car_velocity[2]
				car_velocity[2] = car_velocity_predict

				car_position[0] = car_position[1]
				car_position[1] = car_position[2]
				car_position[2] = car_position_predict
		
		return 4, i,trajectory ## target vehicle will not change lane in 0.8s
	

def behavior_predict(road_width, road_position, car_width, car_length, car_position, car_velocity, laneId, laneNum, model2, model3, model5, model6, model_c3, model_c8):
	if (laneNum == 4):
		if (ChangeCheck(road_width, road_position, car_width, car_length, car_position, car_velocity)==0):
			if (laneId == 2):
				current_data = np.array([[car_position[0], car_position[1], car_velocity[0], car_velocity[1], 2]])
				predicted_states = model2.predict(current_data)
				predicted_laneid = np.where(predicted_states == 0, 2, 3)
				if (predicted_laneid == 2):
					print(f"Target Vehicle in Lane 2 Will not Change Lane in future 0.8s")
					return 2
				else:
					print(f"Target Vehicle in Lane 2 Will Change to Lane3 in future 0.8s")
					return 3

			if (laneId == 3):
				current_data = np.array([[car_position[0], car_position[1], car_velocity[0], car_velocity[1], 2]])
				predicted_states = model3.predict(current_data)
				predicted_laneid = np.where(predicted_states == 0, 2, 3)
				if (predicted_laneid == 3):
					print(f"Target Vehicle in Lane 3 Will not Change Lane in future 0.8s")
					return 3
				else:
					print(f"Target Vehicle in Lane 3 Will Change to Lane2 in future 0.8s")
					return 2

			if (laneId == 5):
				current_data = np.array([[car_position[0], car_position[1], car_velocity[0], car_velocity[1], 2]])
				predicted_states = model5.predict(current_data)
				predicted_laneid = np.where(predicted_states == 0, 5, 6)
				if (predicted_laneid == 5):
					print(f"Target Vehicle in Lane 5 Will not Change Lane in future 0.8s")
				else:
					print(f"Target Vehicle in Lane 5 Will Change to Lane6 in future 0.8s")

			if (laneId == 6):
				current_data = np.array([[car_position[0], car_position[1], car_velocity[0], car_velocity[1], 2]])
				predicted_states = model6.predict(current_data)
				predicted_laneid = np.where(predicted_states == 0, 5, 6)
				if (predicted_laneid == 6):
					print(f"Target Vehicle in Lane 6 Will not Change Lane in future 0.8s")
				else:
					print(f"Target Vehicle in Lane 6 Will Change to Lane5 in future 0.8s")

		else:
			if (laneId == 2): 	print(f"Target Vehicle in Lane 2 is Changing to Lane 3 Now")
			if (laneId == 3): 	print(f"Target Vehicle in Lane 3 is Changing to Lane 2 Now")
			if (laneId == 5): 	print(f"Target Vehicle in Lane 5 is Changing to Lane 6 Now")
			if (laneId == 6): 	print(f"Target Vehicle in Lane 6 is Changing to Lane 5 Now")


	if (laneNum == 6):
		if (ChangeCheck(road_width, road_position, car_width, car_length, car_position, car_velocity)==0):
			if (laneId == 3):
				current_data = np.array([[car_position[0], car_position[1], car_velocity[0], car_velocity[1], 2]])
				predicted_states = model_c3.predict(current_data)
				predicted_laneid = np.where(predicted_states == 0, 2, 3, 4)
				if (predicted_laneid == 3):
					print(f"Target Vehicle in Lane 2 Will not Change Lane in future 0.8s")
					return 3
				if (predicted_laneid == 2):
					print(f"Target Vehicle in Lane 3 Will Change to Lane2 in future 0.8s")
					return 2
				if (predicted_laneid == 4):
					print(f"Target Vehicle in Lane 3 Will Change to Lane4 in future 0.8s")
					return 4
				
			if (laneId == 8):
				current_data = np.array([[car_position[0], car_position[1], car_velocity[0], car_velocity[1], 2]])
				predicted_states = model_c8.predict(current_data)
				predicted_laneid = np.where(predicted_states == 0, 7, 8, 9)
				if (predicted_laneid == 8):
					print(f"Target Vehicle in Lane 8 Will not Change Lane in future 0.8s")
					return 3
				if (predicted_laneid == 7):
					print(f"Target Vehicle in Lane 8 Will Change to Lane7 in future 0.8s")
					return 7
				if (predicted_laneid == 9):
					print(f"Target Vehicle in Lane 8 Will Change to Lane9 in future 0.8s")
					return 9

		else:
			print(f"Target Vehicle is Changing to Lane 3 Now")


def HMMBehaviorPredict(car_position, car_velocity, laneId, laneNum, model2, model3, model5, model6, model_c3, model_c8):
	if (laneNum == 4):
		if (laneId == 2):
			current_data = np.array([[car_position[0], car_position[1], car_velocity[0], car_velocity[1], 2]])
			predicted_states = model2.predict(current_data)
			predicted_laneid = np.where(predicted_states == 0, 2, 3)
			if (predicted_laneid == 2):
				print(f"Target Vehicle in Lane 2 Will not Change Lane in future 0.8s")
				return 2
			else:
				print(f"Target Vehicle in Lane 2 Will Change to Lane3 in future 0.8s")
				return 3

		if (laneId == 3):
			current_data = np.array([[car_position[0], car_position[1], car_velocity[0], car_velocity[1], 2]])
			predicted_states = model3.predict(current_data)
			predicted_laneid = np.where(predicted_states == 0, 2, 3)
			if (predicted_laneid == 3):
				print(f"Target Vehicle in Lane 3 Will not Change Lane in future 0.8s")
				return 3
			else:
				print(f"Target Vehicle in Lane 3 Will Change to Lane2 in future 0.8s")
				return 2

		if (laneId == 5):
			current_data = np.array([[car_position[0], car_position[1], car_velocity[0], car_velocity[1], 2]])
			predicted_states = model5.predict(current_data)
			predicted_laneid = np.where(predicted_states == 0, 5, 6)
			if (predicted_laneid == 5):
				print(f"Target Vehicle in Lane 5 Will not Change Lane in future 0.8s")
			else:
				print(f"Target Vehicle in Lane 5 Will Change to Lane6 in future 0.8s")

		if (laneId == 6):
			current_data = np.array([[car_position[0], car_position[1], car_velocity[0], car_velocity[1], 2]])
			predicted_states = model6.predict(current_data)
			predicted_laneid = np.where(predicted_states == 0, 5, 6)
			if (predicted_laneid == 6):
				print(f"Target Vehicle in Lane 6 Will not Change Lane in future 0.8s")
			else:
				print(f"Target Vehicle in Lane 6 Will Change to Lane5 in future 0.8s")



	if (laneNum == 6):
		if (laneId == 3):
			current_data = np.array([[car_position[0], car_position[1], car_velocity[0], car_velocity[1], 2]])
			predicted_states = model_c3.predict(current_data)
			predicted_laneid = np.where(predicted_states == 0, 2, 3, 4)
			if (predicted_laneid == 3):
				print(f"Target Vehicle in Lane 2 Will not Change Lane in future 0.8s")
				return 3
			if (predicted_laneid == 2):
				print(f"Target Vehicle in Lane 3 Will Change to Lane2 in future 0.8s")
				return 2
			if (predicted_laneid == 4):
				print(f"Target Vehicle in Lane 3 Will Change to Lane4 in future 0.8s")
				return 4
			
		if (laneId == 8):
			current_data = np.array([[car_position[0], car_position[1], car_velocity[0], car_velocity[1], 2]])
			predicted_states = model_c8.predict(current_data)
			predicted_laneid = np.where(predicted_states == 0, 7, 8, 9)
			if (predicted_laneid == 8):
				print(f"Target Vehicle in Lane 8 Will not Change Lane in future 0.8s")
				return 3
			if (predicted_laneid == 7):
				print(f"Target Vehicle in Lane 8 Will Change to Lane7 in future 0.8s")
				return 7
			if (predicted_laneid == 9):
				print(f"Target Vehicle in Lane 8 Will Change to Lane9 in future 0.8s")
				return 9


def HMMBehaviorPredict4Lane(car_position, car_velocity, laneId, model2, model3, model5, model6):
	if (laneId == 2):
		current_data = np.array([[car_position[0][0], car_velocity[0][0], 2],
                         [car_position[1][0], car_velocity[1][0], 2],
                         [car_position[2][0], car_velocity[2][0], 2]])
		predicted_states = model2.predict(current_data)
		predicted_laneid = np.where(predicted_states == 0, 2, 3)
		if (predicted_laneid == 2):
			print(f"Target Vehicle in Lane 2 Will not Change Lane in future 0.8s")
			return 2
		else:
			print(f"Target Vehicle in Lane 2 Will Change to Lane3 in future 0.8s")
			return 3

	if (laneId == 3):
		current_data = np.array([[car_position[0][0], car_velocity[0][0], 3],
                         [car_position[1][0], car_velocity[1][0], 3],
                         [car_position[2][0], car_velocity[2][0], 3]])
		predicted_states = model3.predict(current_data)
		predicted_laneid = np.where(predicted_states == 0, 2, 3)
		if (predicted_laneid == 3):
			print(f"Target Vehicle in Lane 3 Will not Change Lane in future 0.8s")
			return 3
		else:
			print(f"Target Vehicle in Lane 3 Will Change to Lane2 in future 0.8s")
			return 2

	if (laneId == 5):
		current_data = np.array([[car_position[0][0], car_velocity[0][0], 5],
                         [car_position[1][0], car_velocity[1][0], 5],
                         [car_position[2][0], car_velocity[2][0], 5]])
		predicted_states = model5.predict(current_data)
		predicted_laneid = np.where(predicted_states == 0, 5, 6)
		if (predicted_laneid == 5):
			print(f"Target Vehicle in Lane 5 Will not Change Lane in future 0.8s")
		else:
			print(f"Target Vehicle in Lane 5 Will Change to Lane6 in future 0.8s")

	if (laneId == 6):
		current_data = np.array([[car_position[0][0], car_velocity[0][0], 6],
                         [car_position[1][0], car_velocity[1][0], 6],
                         [car_position[2][0], car_velocity[2][0], 6]])
		predicted_states = model6.predict(current_data)
		predicted_laneid = np.where(predicted_states == 0, 5, 6)
		if (predicted_laneid == 6):
			print(f"Target Vehicle in Lane 6 Will not Change Lane in future 0.8s")
		else:
			print(f"Target Vehicle in Lane 6 Will Change to Lane5 in future 0.8s")


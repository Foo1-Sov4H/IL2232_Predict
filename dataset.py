import numpy as np
import pandas as pd

def StateDefine(car_length,car_width, x, y, xVelocity, yVelocity, laneId):
	
	# output :: distance2line_left, distance2line_right, result_left, result_right
	## distance2line_left :: The shortest distance between the target vehicle and the left side of the lane
	## distance2line_right :: The shortest distance between the target vehicle and the right side of the lane
	## result_left :: 1 - The top of car is closer to the left side of the lane
			## 2 - The bot of car is closer to the left side of the lane
	## result_right :: 1 - The top of car is closer to the right side of the lane
			## 2 - The bot of car is closer to the right side of the lane
	v = (xVelocity ** 2 + yVelocity ** 2) ** 0.5
	direction_vector = [xVelocity/v, yVelocity/v]
	half_length = car_length / 2
	half_width = car_width / 2
	position_top_left_x = x + half_length * direction_vector[0] - half_width * direction_vector[1] 
	position_top_left_y = y + half_length * direction_vector[1] + half_width * direction_vector[0]
	position_top_right_x = x + half_length * direction_vector[0] + half_width * direction_vector[1]
	position_top_right_y = y + half_length * direction_vector[1] - half_width * direction_vector[0]
	position_bot_left_x = x - half_length * direction_vector[0] - half_width * direction_vector[1]
	position_bot_left_y = y - half_length * direction_vector[1] + half_width * direction_vector[0]
	position_bot_right_x = x - half_length * direction_vector[0] + half_width * direction_vector[1]
	position_bot_right_y = y - half_length * direction_vector[1] - half_width * direction_vector[0]
	#print("(x,y):",x,y)
	#print("position_top_left (x,y):",position_top_left_x,position_top_left_y)
	#print("position_top_right (x,y):",position_top_right_x,position_top_right_y)
	#print("position_bot_left (x,y):",position_bot_left_x,position_bot_left_y)
	#print("position_bot_right (x,y):",position_bot_right_x,position_bot_right_y)


	if (laneId == 2):
		if all(var <= 11.6 for var in (position_top_left_y, position_top_right_y, position_bot_right_y, position_bot_left_y)):
			state = 2
		else:
			state = 3
	if (laneId == 3):
		if all(11.6 <= var for var in (position_top_left_y, position_top_right_y, position_bot_right_y, position_bot_left_y)):
			state = 4
		else:
			state = 3

	if (laneId == 5):
		if all(var <= 24 for var in (position_top_left_y, position_top_right_y, position_bot_right_y, position_bot_left_y)):
			state = 5
		else:
			state = 6

	if (laneId == 6):
		if all(24 <= var for var in (position_top_left_y, position_top_right_y, position_bot_right_y, position_bot_left_y)):
			state = 7
		else:
			state = 6

	return state

	
#road_width = 3.5
#road_position = [[2, 9.85],[3, 13.35],[5, 22.25],[6, 25.75]]
#2,3 : y = 11.6
#5,6 : y = 24
#StateDefine(4.85, 2.12 ,362.26, 21.68,40.85,0.0,5 )
df = pd.read_csv('highD-dataset-v1.0/data/01_tracks.csv')

df['state'] = df.apply(lambda row: StateDefine(row['width'], row['height'], row['x'], row['y'], row['xVelocity'], row['yVelocity'], row['laneId']), axis=1)

output_csv_path = 'new_state_file.csv'
df.to_csv(output_csv_path, index=False)
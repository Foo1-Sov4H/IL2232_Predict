import pandas as pd
from hmmlearn import hmm
import numpy as np

df = pd.read_csv('training_dataset.csv')

grouped_data = df.groupby(['id', df.groupby('id')['laneId'].transform('first')])

data_dict = {}

for group_name, group_data in grouped_data:
    data_dict[group_name] = group_data

selected_groups2 = grouped_data.filter(lambda x: x['laneId'].iloc[0] == 2)
selected_groups3 = grouped_data.filter(lambda x: x['laneId'].iloc[0] == 3)
selected_groups5 = grouped_data.filter(lambda x: x['laneId'].iloc[0] == 5)
selected_groups6 = grouped_data.filter(lambda x: x['laneId'].iloc[0] == 6)

train_data2 = selected_groups2[['x', 'y', 'xVelocity', 'yVelocity', 'state']].values
train_data3 = selected_groups3[['x', 'y', 'xVelocity', 'yVelocity', 'state']].values
train_data5 = selected_groups5[['x', 'y', 'xVelocity', 'yVelocity', 'state']].values
train_data6 = selected_groups6[['x', 'y', 'xVelocity', 'yVelocity', 'state']].values


# HMM model for lane 2 & 3

df = pd.read_csv('train_dataset2c.csv')

train_data2c = df[['x', 'y', 'xVelocity', 'yVelocity', 'state']].values

model2c = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=10000)

model2c.fit(train_data2c)

print("Training Finish!")
print(model2c.means_)

# Test for Vehicle on lane 2

right = [0, 0, 0, 0, 0]
error = [0, 0, 0, 0, 0]
for i in range(1000):
    # 'x', 'y', 'xVelocity', 'yVelocity', 'state'
    current_data = [train_data2[i+2]]
    history_data1 = [train_data2[i]]
    history_data2 = [train_data2[i+1]]
    #print(history_data1,history_data2,current_data,)
    for j in range(5):        
        predicted_state = model2c.predict(current_data)
        #print('predict state is:',predicted_state)
        posterior_probs = model2c.predict_proba(current_data)
        most_probable_state = np.argmax(posterior_probs)
        predicted_observation = model2c.means_[most_probable_state]
        if (predicted_state == [3]):
            predict_result = current_data[0][4] 
        else:
            predict_result = predicted_observation[4]
        if(predict_result==int(train_data2[i+j+3][4])):
            right[j] = right[j] + 1
        else:
            error[j] = error[j] + 1
        
        next_vx = current_data[0][2] + ((current_data[0][2] - history_data2[0][2]) - (history_data2[0][2] - history_data1[0][2]))
        next_vy = current_data[0][3] + ((current_data[0][3] - history_data2[0][3]) - (history_data2[0][3] - history_data1[0][3]))
        next_x = 0.04 * 0.5 * (next_vx + current_data[0][2]) + current_data[0][0]
        next_y = 0.04 * 0.5 * (next_vy + current_data[0][3]) + current_data[0][1]

        current_data = [[next_x, next_y, next_vx, next_vy, predict_result]]
        history_data1 = history_data2
        history_data2 = current_data
        print('Predict result is: ', current_data)
        print('Real result is:', train_data2[i+j+3])
        
print('Right Num:', right, 'Error Num:', error)

# Test for Vehicle on lane 3

right = [0, 0, 0, 0, 0]
error = [0, 0, 0, 0, 0]
for i in range(1000):
    # 'x', 'y', 'xVelocity', 'yVelocity', 'state'
    current_data = [train_data3[i+2]]
    history_data1 = [train_data3[i]]
    history_data2 = [train_data3[i+1]]
    #print(history_data1,history_data2,current_data,)
    for j in range(5):        
        predicted_state = model2c.predict(current_data)
        #print('predict state is:',predicted_state)
        posterior_probs = model2c.predict_proba(current_data)
        most_probable_state = np.argmax(posterior_probs)
        predicted_observation = model2c.means_[most_probable_state]
        if (predicted_state == [3]):
            predict_result = current_data[0][4] 
        else:
            predict_result = predicted_observation[4]
        if(predict_result==int(train_data3[i+j+3][4])):
            right[j] = right[j] + 1
        else:
            error[j] = error[j] + 1
        
        next_vx = current_data[0][2] + ((current_data[0][2] - history_data2[0][2]) - (history_data2[0][2] - history_data1[0][2]))
        next_vy = current_data[0][3] + ((current_data[0][3] - history_data2[0][3]) - (history_data2[0][3] - history_data1[0][3]))
        next_x = 0.04 * 0.5 * (next_vx + current_data[0][2]) + current_data[0][0]
        next_y = 0.04 * 0.5 * (next_vy + current_data[0][3]) + current_data[0][1]

        current_data = [[next_x, next_y, next_vx, next_vy, predict_result]]
        history_data1 = history_data2
        history_data2 = current_data
        print('Predict result is: ', current_data)
        print('Real result is:', train_data3[i+j+3])

print('Right Num:', right, 'Error Num:', error)
# Right Num: [998, 996, 994, 992, 990] Error Num: [2, 4, 6, 8, 10]


# HMM model for lane 5 & 6


df = pd.read_csv('train_dataset6c.csv')

train_data6c = df[['x', 'y', 'xVelocity', 'yVelocity', 'state']].values
print(train_data6c)
model6c = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=10000)

model6c.fit(train_data6c)

print("Training Finish!")
print(model6c.means_)


right = [0, 0, 0, 0, 0]
error = [0, 0, 0, 0, 0]
for i in range(1000):
    # 'x', 'y', 'xVelocity', 'yVelocity', 'state'
    current_data = [train_data3[i+2]]
    history_data1 = [train_data3[i]]
    history_data2 = [train_data3[i+1]]
    #print(history_data1,history_data2,current_data,)
    for j in range(5):        
        predicted_state = model2c.predict(current_data)
        #print('predict state is:',predicted_state)
        posterior_probs = model2c.predict_proba(current_data)
        most_probable_state = np.argmax(posterior_probs)
        predicted_observation = model2c.means_[most_probable_state]
        if (predicted_state == [3]):
            predict_result = current_data[0][4] 
        else:
            predict_result = predicted_observation[4]
        if(predict_result==int(train_data3[i+j+3][4])):
            right[j] = right[j] + 1
        else:
            error[j] = error[j] + 1
        
        next_vx = current_data[0][2] + ((current_data[0][2] - history_data2[0][2]) - (history_data2[0][2] - history_data1[0][2]))
        next_vy = current_data[0][3] + ((current_data[0][3] - history_data2[0][3]) - (history_data2[0][3] - history_data1[0][3]))
        next_x = 0.04 * 0.5 * (next_vx + current_data[0][2]) + current_data[0][0]
        next_y = 0.04 * 0.5 * (next_vy + current_data[0][3]) + current_data[0][1]

        current_data = [[next_x, next_y, next_vx, next_vy, predict_result]]
        history_data1 = history_data2
        history_data2 = current_data
        print('Predict result is: ', current_data)
        print('Real result is:', train_data3[i+j+3])

        
print('Right Num:', right, 'Error Num:', error)


right = [0, 0, 0, 0, 0]
error = [0, 0, 0, 0, 0]
for i in range(1000):
    # 'x', 'y', 'xVelocity', 'yVelocity', 'state'
    current_data = [train_data5[i+2]]
    history_data1 = [train_data5[i]]
    history_data2 = [train_data5[i+1]]
    #print(history_data1,history_data2,current_data,)
    for j in range(5):        
        predicted_state = model6c.predict(current_data)
        #print('predict state is:',predicted_state)
        posterior_probs = model6c.predict_proba(current_data)
        most_probable_state = np.argmax(posterior_probs)
        predicted_observation = model6c.means_[most_probable_state]
        if (predicted_state == [2]):
            predict_result = current_data[0][4] 
        else:
            predict_result = predicted_observation[4]
        if(predict_result==int(train_data5[i+j+3][4])):
            right[j] = right[j] + 1
        else:
            error[j] = error[j] + 1
        
        next_vx = current_data[0][2] + ((current_data[0][2] - history_data2[0][2]) - (history_data2[0][2] - history_data1[0][2]))
        next_vy = current_data[0][3] + ((current_data[0][3] - history_data2[0][3]) - (history_data2[0][3] - history_data1[0][3]))
        next_x = 0.04 * 0.5 * (next_vx + current_data[0][2]) + current_data[0][0]
        next_y = 0.04 * 0.5 * (next_vy + current_data[0][3]) + current_data[0][1]

        current_data = [[next_x, next_y, next_vx, next_vy, predict_result]]
        history_data1 = history_data2
        history_data2 = current_data
        print('Predict result is: ', current_data)
        print('Real result is:', train_data5[i+j+3])

        
print('Right Num:', right, 'Error Num:', error)


right = [0, 0, 0, 0, 0]
error = [0, 0, 0, 0, 0]
for i in range(1000):
    # 'x', 'y', 'xVelocity', 'yVelocity', 'state'
    current_data = [train_data6[i+2]]
    history_data1 = [train_data6[i]]
    history_data2 = [train_data6[i+1]]
    for j in range(5):        
        predicted_state = model6c.predict(current_data)
        #print('predict state is:',predicted_state)
        posterior_probs = model6c.predict_proba(current_data)
        most_probable_state = np.argmax(posterior_probs)
        predicted_observation = model6c.means_[most_probable_state]
        if (predicted_state == [2]):
            predict_result = current_data[0][4] 
        else:
            predict_result = predicted_observation[4]
        if(predict_result==int(train_data6[i+j+3][4])):
            right[j] = right[j] + 1
        else:
            error[j] = error[j] + 1
        
        next_vx = current_data[0][2] + ((current_data[0][2] - history_data2[0][2]) - (history_data2[0][2] - history_data1[0][2]))
        next_vy = current_data[0][3] + ((current_data[0][3] - history_data2[0][3]) - (history_data2[0][3] - history_data1[0][3]))
        next_x = 0.04 * 0.5 * (next_vx + current_data[0][2]) + current_data[0][0]
        next_y = 0.04 * 0.5 * (next_vy + current_data[0][3]) + current_data[0][1]

        current_data = [[next_x, next_y, next_vx, next_vy, predict_result]]
        history_data1 = history_data2
        history_data2 = current_data
        print('Predict result is: ', current_data)
        print('Real result is:', train_data6[i+j+3])

        
print('Right Num:', right, 'Error Num:', error)
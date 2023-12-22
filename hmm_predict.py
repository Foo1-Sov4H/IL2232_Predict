import pandas as pd
from hmmlearn import hmm
import numpy as np
from collections import Counter


df = pd.read_csv('training_dataset.csv')

grouped_data = df.groupby(['id', df.groupby('id')['laneId'].transform('first')])

data_dict = {}

for group_name, group_data in grouped_data:
    data_dict[group_name] = group_data

selected_groups_forward = grouped_data.filter(lambda x: x['laneId'].iloc[0] in [2, 3])
#selected_groups_backward = grouped_data.filter(lambda x: x['laneId'].iloc[0] in [5, 6])


train_data_forward = selected_groups_forward[['x', 'y', 'xVelocity', 'yVelocity', 'state']].values
#train_data_backward = selected_groups_backward[['x', 'y', 'xVelocity', 'yVelocity', 'state']].values


#model_backward = hmm.GaussianHMM(n_components=6, covariance_type="full", n_iter=1000)
model_forward = hmm.GaussianHMM(n_components=12, covariance_type="full", n_iter=1000)

#model_backward.fit(train_data_backward)
model_forward.fit(train_data_forward)

print("Training Finish!")

### lane2  state 2
### lane2->3 state 3
### lane3  state 4
current_data = [[162.75,9.39,-32.04,0.00,2],
                [161.58,9.39,-32.06,0.00,2],
                [160.36,9.39,-32.07,0.00,2]]

for i in range(5):
    predicted_states = model_forward.predict(current_data)
    matrix = model_forward.means_
    print(matrix)
    last_column = matrix[:, -1].astype(int)
    most_common_value = Counter(last_column).most_common(1)[0][0]
    filtered_matrix = matrix[last_column == most_common_value]
    vx_predict = np.mean(filtered_matrix[:, 2])
    vy_predict = np.mean(filtered_matrix[:, 3])
    x_predict = 0.04 * 0.5 * (vx_predict + current_data[2][2]) + current_data[2][0]
    y_predict = 0.04 * 0.5 * (vy_predict + current_data[2][3]) + current_data[2][1]
    print("Pridect Result:" , x_predict, y_predict, vx_predict, vy_predict, most_common_value)
    current_data = [current_data[1], current_data[2],
                    [x_predict, y_predict, vx_predict, vy_predict, most_common_value]]
## 3 3 3 3 3 2 2 4 4 4 
##Result:
# hidden state mean:
#[[ 2.02141354e+02  1.33109540e+01 -3.29552675e+01 -7.55215287e-03  4.00000000e+00]
# [ 2.07710547e+02  9.80253077e+00 -3.31883583e+01 -2.79903778e-02  2.00000000e+00]
# [ 2.02745861e+02  1.20681786e+01 -3.27832012e+01 -3.09041130e-02  3.00000000e+00]
# [ 2.99652056e+02  9.01791648e+00 -2.43437672e+01  2.29993836e-02  3.00000000e+00]
# [ 2.09923077e+02  9.44613928e+00 -2.60526823e+01 -3.77459548e-03  2.00000000e+00]
# [ 8.31976240e+01  8.98230760e+00 -2.46113813e+01 -2.76846841e-02  3.00000000e+00]]




# Want to predict the results after 0.8s
# Our dataset is 0.04s per frame 

##### method 1:
######### The prediction result after 0.8s is obtained by iterating 20 times.
######### Problem: The vehicle's position has been obtained during this iterative process, and there is a considerable error.

##### method 2:
######### The dataset to training can be changed to (pick i = 20 x j + k) to train 
######### Problem: data is not enough, some data group will only have 5 data.
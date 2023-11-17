# IL2232_MPC_Predict

actr.PredictVelocity:
  input: v1, v2, v3
    v1,v2: historical data
	  v3: current data
  output: predict_v, execution_time


actr.PredictLocationLine:
  input: x1 , y1 , x2 , y2 , x3 , y3 , v3 , predict_v
    (x1,y1) (x2,y2): historical data
	  (x3,y3,v3): current data
    predict_v: result from PredictVelocity
  output: (predict_x, predict_y)  execution_time



actr.PredictLocationCircle:
  input: x1 , y1 , x2 , y2 , x3 , y3 , v3 , predict_v
    (x1,y1) (x2,y2): historical data
	  (x3,y3,v3): current data
    predict_v: result from PredictVelocity
  output: (predict_x, predict_y) , execution_time

Before using PredictLocation, we need to determine the current state of the motion and this is done by comparing the slopes.

# IL2232_MPC_Predict

actr.PredictVelocity:
\
  input: v1, v2, v3
\
    v1,v2: historical data    
\
    v3: current data
\
  output: predict_v, execution_time
\


actr.Judgment:
\
  input: x1 , y1 , x2 , y2 , x3 , y3
  \
    (x1,y1) (x2,y2): historical data
    \
    (x3,y3): current data
    \
  output: 0 or 1
  \
    0 : Circle
    \
    1 : Line
    \


actr.PredictLocationLine:
\
  input: x1 , y1 , x2 , y2 , x3 , y3 , v3 , predict_v
  \
    (x1,y1) (x2,y2): historical data
    \
    (x3,y3,v3): current data
    \
    predict_v: result from PredictVelocity
    \
  output: (predict_x, predict_y)  execution_time
  \




actr.PredictLocationCircle:
\
  input: x1 , y1 , x2 , y2 , x3 , y3 , v3 , predict_v
  \
    (x1,y1) (x2,y2): historical data
    \
	  (x3,y3,v3): current data
   \
    predict_v: result from PredictVelocity
    \
  output: (predict_x, predict_y) , execution_time
  \

Before using PredictLocation, we need to determine the current state by actr.Judgment function.

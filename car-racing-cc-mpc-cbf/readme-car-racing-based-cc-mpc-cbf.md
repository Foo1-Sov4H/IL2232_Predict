# readme-car-racing-based-cc-mpc-cbf

forked from [https://github.com/HybridRobotics/Car-Racing](https://github.com/HybridRobotics/Car-Racing)

# How to run

Download the repo.

Install anaconda. (I use Anaconda3 2022.10 Python 3.9.13 64-bit. windows)

creating a new conda environment:

```python
conda env create -f environment.yml
```

activate the environment before run

```python
conda activate car-racing
```

install dependency:

```python
pip install -e .
```

Run MPC_CBF test:

```python
python car_racing/tests/mpccbf_test.py --track-layout l_shape --simulation --plotting --animation
```

possible layouts:

- `l_shape`
- `m_shape`
- `goggle`
- `ellipse`

With the -plotting flag, the profile of the ego vehicle and the vehicle trajectory map will be generated and showed after the program finished. It won’t be saved to local unless you click the save button.

The ego vehicle state profile contains four subgraph, they are:

- longitudinal velocity Vx
- lateral velocity Vy
- heading angle error between the vehicle and path e_phi
- deviation distance from the path e_yt

With the -animation flag, the GIF animation will be generated, and stored under /car-racing/media/animation.

It won’t replace the old GIF file in this directory. So remove the old GIF file before run.

Refer to the newest guide if encounter other problems:  [https://github.com/HybridRobotics/Car-Racing?tab=readme-ov-file#installation](https://github.com/HybridRobotics/Car-Racing?tab=readme-ov-file#installation)

# CC-MPC-CBF Code Guide

Implementation of chance-constrained MPC-CBF:

\car-racing\car_racing\control\[control.py](http://control.py/)

check method mpccbf():

chance-constraint related configurations:

```python
APPLY_CHANCE_CONSTRAINT = True # True to enable chance-constraint. False to disable
sigma = 0.7 # noise
sigma_squared = sigma * sigma
smallS_gamma = 0.9 # hyperparameter used in smallS
zeta = 1
```

The following code implemented the chance constrained mpc cpf. Refer to paper [https://arxiv.org/abs/2304.01639](https://arxiv.org/abs/2304.01639)

```python
# Chance constraint
W = ca.diag([1]) # diagonal matrix
W_t = ca.transpose(W)
T = [0, 0, 0, 0, 1, 0]
T = ca.transpose(T)
M_B = ca.mtimes(T, mpc_cbf_param.matrix_B) # defined in /data/sys/LTI/matrix_B.csv
M_A = ca.mtimes(T, mpc_cbf_param.matrix_A) # defined in /data/sys/LTI/matrix_A.csv
B_t = ca.mtimes(ca.transpose(mpc_cbf_param.matrix_B), ca.transpose(T))
A_t = ca.mtimes(ca.transpose(mpc_cbf_param.matrix_A), ca.transpose(T))
# logging.info("mpc_cbf_param.matrix_A:")
# logging.info(mpc_cbf_param.matrix_A)
# logging.info("mpc_cbf_param.matrix_B:")
# logging.info(mpc_cbf_param.matrix_B)
phy = ca.mtimes( ca.mtimes(B_t, W), M_B )
logging.info("phy:") # Refer to Moving Obstacle Collision Avoidance via Chance-Constrained MPC with CBF - III. CHANCE CONSTRAINED MPC-CBF
logging.info(phy)

logging.info("bigH:")
bigH = 4 * sigma_squared * ca.mtimes( ca.mtimes(B_t, W_t), ca.mtimes(W, M_B) )
logging.info(bigH)
logging.info(bigH)

logging.info("smallM:")
smallM_gW = ca.mtimes(B_t, W)
smallM = ca.mtimes( smallM_gW, (ca.mtimes(M_A, xvar[4, i]) - obs_traj[4, i]) )
logging.info(smallM)

logging.info("smallN:")
f_x_ki = ca.mtimes(M_A, xvar[4, i])
smallN = 4 * sigma_squared * ca.mtimes( ca.mtimes( ca.mtimes(B_t, W_t), W ), (f_x_ki - obs_traj[4, i]) )
logging.info(smallN)

logging.info("smallS:")
minus = f_x_ki - obs_traj[4, i]
smallS_term1 = ca.norm_fro(minus) # Frobenius norm
smallS_term2 = sigma_squared * ca.trace(W)

minus2 = xvar[4, i] - obs_traj[4, i]
h_term = ca.mtimes( ca.mtimes(ca.transpose(minus2), W), minus2 ) - 1
smallS_term3 = (1-smallS_gamma) * h_term

smallS = smallS_term1 + smallS_term2 - smallS_term3 - 1
logging.info(smallS)

logging.info("smallD f_x_ki:")
logging.info(f_x_ki)
logging.info("smallD obs_traj[4, i]:")
logging.info(obs_traj[4, i])
minus3 = sigma * ca.mtimes(W, (f_x_ki - obs_traj[4, i]))
logging.info("smallD minus3:")
logging.info(minus3)
smallD_term1 =  4 * ca.norm_fro(minus3) # Frobenius norm
logging.info("smallD smallD_term1:")
logging.info(smallD_term1)
smallD_term2 = 2 * sigma_squared * sigma_squared * ca.trace(ca.mtimes(W_t, W))
logging.info("smallD smallD_term2:")
logging.info(smallD_term2)
smallD = smallD_term1 + smallD_term2
logging.info("smallD:")
logging.info(smallD)

u_ki_t = ca.transpose(uvar[:, i])
erfinv_result = sp_sp.erfinv(2 * ca.sqrt(sigma_squared) - 1) # erf⁻¹(2*sigma-1)
c_sigma = math.sqrt(2) * erfinv_result
in_sqrt_1 = ca.mtimes( ca.mtimes(u_ki_t, bigH), uvar[:, i])
in_sqrt_2 =  ca.mtimes(2 * ca.transpose(smallN), uvar[:, i]) + smallD
chance_constraint_line1 = c_sigma * ca.sqrt( in_sqrt_1 + in_sqrt_2)
line2_term1 = ca.mtimes( ca.mtimes(u_ki_t, phy), uvar[:, i])
line2_term2 = ca.mtimes(2 * ca.transpose(smallM), uvar[:, i]) + smallS
chance_constraint_line2 = line2_term1 + line2_term2
chance_constraint_line1_flat = ca.vec(chance_constraint_line1) # flatten
chance_constraint_line2_flat = ca.vec(chance_constraint_line2)
```

# Change simulation environment

in file \car-racing\car_racing\tests\mpccbf_test.py

It’s possible to change ego vehicle speed:

```python
mpc_cbf_param = base.MPCCBFRacingParam(vt=0.8)
```

and setup surrounding cars:

```python
speed = 0.2 # 0.2 by default. target vehicle spped
carsMap = { # (distribution, offsetFromCenterLine, speed)
    "car1": (4, 0, speed),
    "car2": (20, -0.1, speed),
    "car3": (8, 0.1, speed),
    "car4": (6, -0, speed),
    "car5": (10, 0.1, speed),
    "car6": (14, -0, speed)
}
```

carsMap recorded the parameter of each target vehicle. The first parameter is the lateral position distribution, and the second one is the offset from center line. The third one is speed.
To change the placement of target vehicles, only need to add/remove from the carsMap.
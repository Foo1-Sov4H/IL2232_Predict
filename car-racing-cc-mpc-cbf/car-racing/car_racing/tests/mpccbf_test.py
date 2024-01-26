import pickle
import sympy as sp
import numpy as np
from racing import offboard
from utils import base, racing_env
from utils.constants import *


def racing(args):
    track_layout = args["track_layout"]
    track_spec = np.genfromtxt("data/track_layout/" + track_layout + ".csv", delimiter=",")
    if args["simulation"]:
        track = racing_env.ClosedTrack(track_spec, track_width=1.0)
        # setup ego car
        ego = offboard.DynamicBicycleModel(name="ego", param=base.CarParam(edgecolor="black"), system_param = base.SystemParam())
        mpc_cbf_param = base.MPCCBFRacingParam(vt=0.8)
        ego.set_state_curvilinear(np.zeros((X_DIM,)))
        ego.set_state_global(np.zeros((X_DIM,)))
        ego.start_logging()
        ego.set_ctrl_policy(offboard.MPCCBFRacing(mpc_cbf_param, ego.system_param))
        ego.ctrl_policy.set_timestep(0.1)
        ego.set_track(track)
        ego.ctrl_policy.set_track(track)
        # setup surrounding cars
        speed = 0.2 # 0.2 by default. target vehicle spped
        carsMap = { # (distribution, offsetFromCenterLine, speed)
            "car1": (4, 0, speed),
            "car2": (8, 0, speed),
            "car3": (6, -0, speed),
            "car4": (10, 0, speed),
            "car5": (14, -0, speed)
        }

        car_num = len(carsMap)

        t_symbol = sp.symbols("t")
        cars = []
        for i in range(1, car_num + 1):
            car_name = f"car{i}"
            car = create_car(car_name, track, t_symbol, carsMap)
            cars.append(car)

        # setup simulation
        simulator = offboard.CarRacingSim()
        simulator.set_timestep(0.1)
        simulator.set_track(track)
        simulator.add_vehicle(ego)
        ego.ctrl_policy.set_racing_sim(simulator)
        for car in cars:
            simulator.add_vehicle(car)

        simulator.sim(sim_time=50.0)
        with open("data/simulator/racing.obj", "wb") as handle:
            pickle.dump(simulator, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("data/simulator/racing.obj", "rb") as handle:
            simulator = pickle.load(handle)
    if args["plotting"]:
        simulator.plot_simulation()
        simulator.plot_state("ego")
    if args["animation"]:
        simulator.animate(filename="racing", imagemagick=True)

def create_car(name, track, t_symbol, carsMap):
    car = offboard.NoDynamicsModel(name, param=base.CarParam(edgecolor="orange"))
    car.set_track(track)
    car.set_state_curvilinear_func(t_symbol, carsMap[name][2] * t_symbol + carsMap[name][0], carsMap[name][1] + 0.0 * t_symbol)
    car.start_logging()
    return car

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--plotting", action="store_true")
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--track-layout", type=str)
    args = vars(parser.parse_args())
    racing(args)

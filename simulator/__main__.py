import simpy
import pandas as pd
from tqdm import tqdm 

import os
import logging
from datetime import datetime

from config import DATA_PATH, VERBOSITY
from simulator.position_generator import UniformCoordinateGenerator
from simulator.dispatcher import RideSharingSimulation


if __name__ == "__main__":

    simulation_data = []

    logging.basicConfig(
        level=VERBOSITY, 
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_simulation.csv"

    for simulation_id in tqdm(range(0, 10)): 
        env = simpy.Environment()

        simulation = (
            RideSharingSimulation(
                env=env, 
                id=float(simulation_id), 
                num_cars=20, 
                num_users=0, 
                inflow_rate=22, 
                map_size=(10, 10), 
                simulation_time=4000, 
                coordinate_generator=UniformCoordinateGenerator(10,10)
            )
        )
        simulation.run()
        simulation.load_simulation_data()
        simulation_data = pd.DataFrame(simulation.user_data)
        simulation_data.to_csv(DATA_PATH / filename, mode='a', index=False, header=not os.path.exists(DATA_PATH / filename))

    print(f"Data Saved in {filename}")

    #####

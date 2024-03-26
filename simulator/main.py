import numpy as np
import simpy
import pandas as pd
from tqdm import tqdm 

import random
import os
import logging
from datetime import datetime
from dataclasses import dataclass
from abc import ABC

from simulator.solvers import MILPSolver
from utils import timeit
from config import DATA_PATH, VERBOSITY
from simulator.position_generator import UniformCoordinateGenerator


simulation_data = []


class Dispatcher:
    """Dispatcher class to manage car-user assignments."""
    def __init__(self, env, users, available_cars, solver):
        self.env = env
        self.solver = solver
        self.users = {user.id: user for user in users}
        self.served_users = []
        self.available_cars = {car.id: car for car in available_cars}

    def solve(self, solver) -> float:
        """Solve the assignment problem using the specified solver."""
        optimal_matching = solver.get_optimal_solution(self.available_cars, self.users)
        return optimal_matching

    def add_new_user(self, user):
        """Add a new user to the simulation."""
        self.users.update({user.id: user})
        
        logging.debug(f"User {user.id} joins the simulation at time {self.env.now}")

    def set_user_to_served(self, user):
        """Remove user from the simulation."""
        self.users.pop(user.id, None)
        self.served_users.append(user)

    def set_car_to_busy(self, car):
        """Remove car from set of availabile cars."""
        self.available_cars.pop(car.id, None)

    def set_car_to_free(self, car):
        """Add car to set of available cars."""
        self.available_cars.update({car.id: car})

    def move_car_to_ride_destination(self, car, user):
        """Set car to position of customer's destination."""
        car.coordinates_start = user.coordinates_start

    def assign_car_to_user(self, optimal_solution):
        """Assign cars to users based on the globally optimal assignment."""
        for optimal_match in optimal_solution:
            matched_car_id = optimal_match.get("cars")
            matched_user_id = optimal_match.get("users")
            assignment_cost = optimal_match.get("assignment_cost")
            car = self.available_cars.get(matched_car_id)
            user = self.users.get(matched_user_id)
            
            user.assignment_cost = round(assignment_cost, ndigits=2)
            self.set_car_to_busy(car)
            self.env.process(user.use_car(car))
            self.move_car_to_ride_destination(car, user)
            self.set_user_to_served(user)
            self.set_car_to_free(car)
            
    def dispatch(self):
        """Dispatch cars to users."""
        if not self.available_cars or not self.users:
            return
        
        optimal_solution = self.solve(self.solver)
        self.assign_car_to_user(optimal_solution)


class RideSharingSimulation:
    """Simulation of car sharing system."""
    def __init__(self, env, id, num_cars, num_users, map_size, simulation_time, inflow_rate, coordinate_generator):
        self.env = env
        self.id = id
        self.num_cars = num_cars
        self.num_users = num_users
        self.map_size = map_size
        self.cars = [Car(env, i, self.map_size, capacity=1, coordinate_generator=coordinate_generator) for i in range(num_cars)]
        self.users = [User(env, i, self.map_size, self.verbose, coordinate_generator) for i in range(num_users)]
        self.available_cars = self.cars  # Initially, all cars are available
        self.dispatcher = Dispatcher(env, self.users, self.cars, MILPSolver(decision_variable_type="real", solver_name="cbc"))
        self.simulation_time = simulation_time
        self.expected_customer_inflow = inflow_rate
        self.coordinate_generator = coordinate_generator

    def dispatch_cars(self):
        """Dispatch cars at regular intervals."""
        while True:
            n_new_users = np.random.poisson(self.expected_customer_inflow)

            for _ in range(0, n_new_users):
                self.dispatcher.add_new_user(User(env, self.num_users, self.map_size, self.coordinate_generator))
                self.num_users += 1

            self.dispatcher.dispatch()
            yield self.env.timeout(15)  # Dispatch cars every 1 minute

    @timeit
    def run(self):
        """Run the simulation."""
        # Schedule the dispatcher to run every 1 minute
        self.env.process(self.dispatch_cars())
        # Start the simulation
        self.env.run(until=self.simulation_time) # Run the simulation for 50 time units

    def load_simulation_data(self):

        self.user_data = []
        self.average_waiting_time = np.nanmean([user.waiting_time for user in self.dispatcher.served_users])
        self.average_service_time = np.nanmean([user.service_time for user in self.dispatcher.served_users])
        self.average_order_fullfilment_time = np.nanmean([user.order_fullfilment_time for user in self.dispatcher.served_users])
        self.average_assignment_cost = np.nanmean([user.assignment_cost for user in self.dispatcher.served_users])
        self.n_served_users = len(self.dispatcher.served_users)

        for user in self.dispatcher.served_users:

            if user.served_status is True:
            
                row_user_data={"user_id": user.id, "car_id": user.car_id, "activation_time": user.request_activation_time,
                    "request_completion_time": user.request_completion_time,"waiting_time": user.waiting_time,
                    "service_time": user.service_time, "order_fullfilment_time": user.order_fullfilment_time, 
                    "assignment_cost": user.assignment_cost, "order_is_served": user.served_status, "simulation_id": self.id}
                
            else: 

                waiting_time_upper_bound = round(self.simulation_time - self.request_activation_time, 2)
                order_fullfilment_time_upper_bound = round(self.simulation_time - self.request_activation_time, 2)

                row_user_data={"user_id": user.id, "car_id": user.car_id, "activation_time": user.request_activation_time,
                    "request_completion_time": user.request_completion_time,"waiting_time": waiting_time_upper_bound,
                    "service_time": user.service_time, "order_fullfilment_time": order_fullfilment_time_upper_bound, 
                    "assignment_cost": user.assignment_cost, "order_is_served": user.served_status, "simulation_id": self.id}
            
            self.user_data.append(row_user_data)


# Setup and run the simulation
# simulation_results = {"simulation_id": [], "global_cost": []}
if __name__ == "__main__":

    logging.basicConfig(
        level=VERBOSITY, 
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_simulation.csv"

    for simulation_id in tqdm(range(0, 10)): 
        env = simpy.Environment()

        simulation = RideSharingSimulation(env=env, id=float(simulation_id), num_cars=20, num_users=0, inflow_rate=22, map_size=(10, 10), simulation_time=4000, coordinate_generator=UniformCoordinateGenerator(10,10))
        simulation.run()
        simulation.load_simulation_data()
        simulation_data = pd.DataFrame(simulation.user_data)
        simulation_data.to_csv(DATA_PATH / filename, mode='a', index=False, header=not os.path.exists(DATA_PATH / filename))

    print(f"Data Saved in {filename}")

    #####

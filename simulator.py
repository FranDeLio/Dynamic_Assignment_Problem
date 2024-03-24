import random
import os

from solvers import MILPSolver
from utils import timeit
from config import DATA_PATH

import numpy as np
from scipy.optimize import linear_sum_assignment
import simpy
from scipy.stats import norm
from scipy.spatial import distance
import pandas as pd
from datetime import datetime


simulation_data = []

class Car(simpy.Resource):
    """Representation of a car in the simulation."""
    def __init__(self, env, id, map_size, capacity):
        self.id = id
        self.map_size = map_size
        self.coordinates_start = (
            random.uniform(0, self.map_size[0]),
            random.uniform(0, self.map_size[1]),
        )
        super().__init__(env, capacity)

class User:
    """Representation of a user in the simulation."""
    def __init__(self, env, id, map_size):
        self.env = env
        self.id = id
        self.map_size = map_size
        self.coordinates_start = (
            random.uniform(0, self.map_size[0]),
            random.uniform(0, self.map_size[1]),
        )
        self.coordinates_destination = (
            random.uniform(0, self.map_size[0]),
            random.uniform(0, self.map_size[1]),
        )
        self.request_activation_time = self.env.now
        self.simulation_id = simulation_id
        self.served_status = False
        self.service_start_time = None
        self.waiting_time = np.nan
        self.service_start_time = None
        self.request_completion_time = None
        self.service_time = np.nan
        self.order_fullfilment_time = np.nan

    def to_dict(self):
        return vars(self)

    def save_assignment_cost(self, assignment_cost):
        self.assignment_cost = round(assignment_cost, ndigits=3)

    def use_car(self, car):
        """User action to use a car."""
        print(f"User {self.id} takes car {car.id} at time {self.env.now}")
        self.car_id = car.id
        with car.request() as request:

            yield request
            #note that start to customer and customer to destination are not split in this logic
            self.waiting_time = round(self.env.now - self.request_activation_time, 2)
            self.service_start_time = round(self.env.now, 2)
            yield self.env.timeout(self.assignment_cost)
            self.request_completion_time = round(self.env.now, 2)
            self.service_time = round(self.request_completion_time - self.service_start_time, 2)
            self.order_fullfilment_time = round(self.request_completion_time - self.request_activation_time, 2)
            self.served_status = True
        
        # User returns the car
        print(f"User {self.id} leaves car {car.id} at time {self.env.now}")

class Dispatcher:
    """Dispatcher class to manage car-user assignments."""
    def __init__(self, env, users, available_cars, solver):
        self.env = env
        self.solver = solver
        self.users = {user.id: user for user in users}
        self.served_users = []
        self.available_cars = {car.id: car for car in available_cars}
        self.simulation_data = []
        self.n_served_customers = 0
        self.global_assignment_cost = 0

    def solve(self, solver) -> float:
        """Solve the assignment problem using the specified solver."""
        optimal_matching = solver.get_optimal_solution(self.available_cars, self.users)
        return optimal_matching

    def add_new_user(self, user):
        """Add a new user to the simulation."""
        self.users.update({user.id: user})

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
            
            user.save_assignment_cost(assignment_cost)
            self.global_assignment_cost += assignment_cost
            self.set_car_to_busy(car)
            self.env.process(user.use_car(car))
            self.move_car_to_ride_destination(car, user)
            self.n_served_customers += 1
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
    def __init__(self, env, id, num_cars, num_users, map_size, simulation_time, inflow_rate):
        self.env = env
        self.id = id
        self.num_cars = num_cars
        self.num_users = num_users
        self.map_size = map_size
        self.cars = [Car(env, i, self.map_size, capacity=1) for i in range(num_cars)]
        self.users = [User(env, i, self.map_size) for i in range(num_users)]
        self.available_cars = self.cars  # Initially, all cars are available
        self.dispatcher = Dispatcher(env, self.users, self.cars, MILPSolver(decision_variable_type="real", solver_name="cbc"))
        self.simulation_time = simulation_time
        self.expected_customer_inflow = inflow_rate

    def dispatch_cars(self):
        """Dispatch cars at regular intervals."""
        while True:
            self.dispatcher.dispatch()
            n_new_users = np.random.poisson(self.expected_customer_inflow)

            for _ in range(0, n_new_users):
                self.dispatcher.add_new_user(User(env, self.num_users, self.map_size))
                print(f"User {self.num_users} joins the simulation at time {env.now}")
                self.num_users += 1

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
            
            row_user_data={"user_id": user.id, "car_id": user.car_id, "activation_time": user.request_activation_time,
                "request_completion_time": user.request_completion_time,"waiting_time": user.waiting_time,
                "service_time": user.service_time, "order_fullfilment_time": user.order_fullfilment_time, 
                "assignment_cost": user.assignment_cost, "orders_is_served": user.served_status, "simulation_id": self.id}
        
            self.user_data.append(row_user_data)


if __name__ == "__main__":

    # Setup and run the simulation
    simulation_results = {"simulation_id": [], "global_cost": []}
    filename = f"{DATA_PATH}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_simulation.csv"

    for simulation_id in range(0, 20): 
        env = simpy.Environment()

        simulation = RideSharingSimulation(env=env, id=simulation_id, num_cars=20, num_users=40, inflow_rate=10, map_size=(10, 10), simulation_time=500)
        simulation.run()
        simulation.load_simulation_data()
        simulation_data = pd.DataFrame(simulation.user_data)
        simulation_data.to_csv(filename, mode='a', index=False, header=not os.path.exists(filename))
    
        print(simulation_data)
        print(f"Service time: {simulation.average_service_time}")
        print(f"Waiting time: {simulation.average_waiting_time}")

    print(f"Data Saved in {filename}")

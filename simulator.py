import time
from functools import wraps
import random
import itertools

from solvers import MILPSolver

import numpy as np
from scipy.optimize import linear_sum_assignment
import simpy
from scipy.stats import norm
from scipy.spatial import distance
import pandas as pd

n_served_customers = 0

def timeit(func):
    """Decorator to measure the execution time of a function."""
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie args[0] is self
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper

class Car(simpy.Resource):
    """Representation of a car in the simulation."""
    def __init__(self, env, id, map_size, capacity):
        self.id = id
        self.map_size = map_size
        self.position = (
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
        self.position = (
            random.uniform(0, self.map_size[0]),
            random.uniform(0, self.map_size[1]),
        )
        self.service_time = 0
        self.waiting_time = 0

    def use_car(self, car):
        """User action to use a car."""
        print(f"User {self.id} takes car {car.id} at time {self.env.now}")
        with car.request() as request:

            service_request_time = self.env.now
            yield request
            self.waiting_time = self.env.now - service_request_time
            service_start_time = self.env.now
            yield self.env.timeout(5)
            self.service_time = self.env.now - service_start_time

        # User returns the car
        print(f"User {self.id} leaves car {car.id} at time {self.env.now}")

    def _use_car(self, car):
        self.waiting_time = 69
        self.service_time = 7


class Dispatcher:
    """Dispatcher class to manage car-user assignments."""
    def __init__(self, env, users, available_cars, solver):
        self.env = env
        self.solver = solver
        self.users = {user.id: user for user in users}
        self.available_cars = {car.id: car for car in available_cars}
        self.simulation_data = []

    def solve(self, solver) -> float:
        """Solve the assignment problem using the specified solver."""
        optimal_matching = solver.get_optimal_solution(self.available_cars, self.users)
        return optimal_matching

    def add_new_user(self, user):
        """Add a new user to the simulation."""
        self.users.update({user.id: user})

    def collect_service_data(self, user, car):
        """Collect service data for analysis."""
        data={"user_id": user.id, "car_id": car.id, "service_time": user.service_time, "waiting_time": user.waiting_time}
        print(data)
        self.simulation_data.append(data)

    def dispatch(self):
        """Dispatch cars to users."""
        global n_served_customers

        if not self.available_cars or not self.users:
            return
        
        optimal_solution = self.solve(self.solver)
        # Assign cars to users based on the globally optimal assignment
        for user_id, car_id in optimal_solution.items():
            
            car=self.available_cars.get(car_id)
            user=self.users.get(user_id)
            self.available_cars.pop(car_id, None)
            self.env.process(user.use_car(car))
            #user._use_car(car)
            self.available_cars.update({car_id: car})
            car.position = user.position
            self.collect_service_data(user, car)
            print(f"fff {user.waiting_time}")
            self.users.pop(user_id, None)
            n_served_customers += 1

class CarSharingSimulation:
    """Simulation of car sharing system."""
    def __init__(self, env, num_cars, num_users, map_size, simulation_time):
        self.env = env
        self.num_cars = num_cars
        self.num_users = num_users
        self.map_size = map_size
        self.cars = [Car(env, i, self.map_size, capacity=1) for i in range(num_cars)]
        self.users = [User(env, i, self.map_size) for i in range(num_users)]
        self.available_cars = self.cars  # Initially, all cars are available
        self.dispatcher = Dispatcher(env, self.users, self.cars, MILPSolver(decision_variable_type="real", solver_name="cbc"))
        self.simulation_time = simulation_time

    @timeit
    def run(self):
        """Run the simulation."""
        # Schedule the dispatcher to run every 1 minute
        self.env.process(self.dispatch_cars())
        # Start the simulation
        self.env.run(until=self.simulation_time) # Run the simulation for 50 time units

    def dispatch_cars(self):
        """Dispatch cars at regular intervals."""
        while True:
            self.dispatcher.dispatch()
            n_new_users = np.random.poisson(1)

            for i in range(0, n_new_users):
                self.dispatcher.add_new_user(User(env, self.num_users, self.map_size))
                print(f"User {self.num_users} joins the simulation at time {env.now}")
                self.num_users += 1

            yield self.env.timeout(15)  # Dispatch cars every 1 minute

    def get_simulation_data(self):
        """Get simulation data for analysis."""
        return self.dispatcher.simulation_data 


# Setup and run the simulation
env = simpy.Environment()
car_sharing_sim = CarSharingSimulation(env=env, num_cars=1, num_users=10, map_size=(10, 10), simulation_time=2000)
car_sharing_sim.run()
simulation_data = pd.DataFrame(car_sharing_sim.get_simulation_data()).to_csv("sim_data.csv")
print(f"Customers Served: {n_served_customers}")
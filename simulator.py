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
simulation_data = []

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

    def set_assignment_cost(self, assignment_cost):
        self.assignment_cost = assignment_cost

    def use_car(self, car):
        """User action to use a car."""
        print(f"User {self.id} takes car {car.id} at time {self.env.now}")
        global simulation_data
        with car.request() as request:

            yield request
            waiting_time = self.env.now - self.request_activation_time
            service_start_time = self.env.now
            yield self.env.timeout(5)
            request_completion_time = self.env.now
            service_time = request_completion_time - service_start_time
            order_fullfilment_time = request_completion_time - self.request_activation_time
        
        data={"user_id": self.id, "car_id": car.id, "activation_time": self.request_activation_time,
              "request_completion_time": request_completion_time,"waiting_time": waiting_time,
              "service_time": service_time, "order_fullfilment_time": order_fullfilment_time, 
              "assignment_cost": self.assignment_cost}
        simulation_data.append(data)

        # User returns the car
        print(f"User {self.id} leaves car {car.id} at time {self.env.now}")

    def _test_use_car(self, car):
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

    def _collect_service_data(self, user, car):
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
        for optimal_match in optimal_solution:
            #improve readability, split into different methods to be called within the for loop
            print(optimal_match)
            car_id = optimal_match.get("cars")
            user_id = optimal_match.get("users")
            assignment_cost = optimal_match.get("assignment_cost")

            car = self.available_cars.get(car_id)
            user = self.users.get(user_id)
            self.available_cars.pop(car_id, None)
            user.set_assignment_cost(assignment_cost)
            self.env.process(user.use_car(car))
            #user._test_use_car(car)
            self.available_cars.update({car_id: car})
            car.coordinates_start = user.coordinates_start
            #self._collect_service_data(user, car)
            #print(f"fff {user.waiting_time}")
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
car_sharing_sim = CarSharingSimulation(env=env, num_cars=1, num_users=4, map_size=(10, 10), simulation_time=200)
car_sharing_sim.run()
simulation_data = pd.DataFrame(simulation_data).to_csv("sim_data.csv", index=False)
print(f"Customers Served: {n_served_customers}")
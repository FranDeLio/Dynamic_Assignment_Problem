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

n_served_customers = 0


def timeit(func):
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
    def __init__(self, env, id, map_size, capacity):
        self.id = id
        self.map_size = map_size
        self.position = (
            random.uniform(0, self.map_size[0]),
            random.uniform(0, self.map_size[1]),
        )
        super().__init__(env, capacity)

    def get_distance_to_user(self, user):
        return np.linalg.norm(np.array(self.position) - np.array(user.position))


class User:
    def __init__(self, env, id, map_size):
        self.env = env
        self.id = id
        self.map_size = map_size
        self.position = (
            random.uniform(0, self.map_size[0]),
            random.uniform(0, self.map_size[1]),
        )

    def use_car(self, car):
        # User gets a car and drives for a random amount of time
        print(f"User {self.id} takes car {car.id} at time {self.env.now}")
        with car.request() as request:
            yield request
            yield self.env.timeout(5)

        # User returns the car
        print(f"User {self.id} leaves car {car.id} at time {self.env.now}")


class Dispatcher:
    def __init__(self, env, users, available_cars, solver):
        self.env = env
        #self.users = users
        #self.available_cars = available_cars
        self.solver = solver
        self.users = {user.id: user for user in users}
        self.available_cars = {car.id: car for car in available_cars}

    def solve(self, solver) -> float:
        optimal_matching = solver.get_optimal_solution(self.available_cars, self.users)

        return optimal_matching

    def add_new_user(self, user):
        self.users[user.id]=user

    def dispatch(self):
        # Generate all possible combinations of users and cars
        assignments = list(itertools.product(self.users, self.available_cars))
        global n_served_customers

        # Find the assignment that minimizes total distance
        if (len(self.available_cars) == 0) | (len(self.users)) == 0:

            pass

        else:
        
            optimal_solution = self.solve(self.solver)

            # Assign cars to users based on the globally optimal assignment
            for user_id, car_id in optimal_solution.items():
                
                car=self.available_cars[car_id]
                user=self.users[user_id]
                print((user, car))
                #indexing fails here as car is just an id of car, and there's no .loc to get the correct car
                #easier approach just index self.available cars
                self.available_cars.pop(car_id, None)
                self.env.process(user.use_car(car))
                self.available_cars[car_id]=car
                car.position = user.position
                self.users.pop(user_id, None)
                n_served_customers += 1

            print((len(self.available_cars), len(self.users)))


class CarSharingSimulation:
    def __init__(self, env, num_cars, num_users, map_size, simulation_time):
        self.env = env
        self.num_cars = num_cars
        self.num_users = num_users
        self.map_size = map_size
        self.cars = [Car(env, i, self.map_size, capacity=1) for i in range(num_cars)]
        self.users = [User(env, i, self.map_size) for i in range(num_users)]
        self.available_cars = self.cars  # Initially, all cars are available
        self.dispatcher = Dispatcher(env, self.users, self.cars, MILPSolver())
        self.simulation_time = simulation_time

    @timeit
    def run(self):
        # Schedule the dispatcher to run every 1 minute
        self.env.process(self.dispatch_cars())

        # Start the simulation
        self.env.run(until=self.simulation_time)  # Run the simulation for 50 time units

    def dispatch_cars(self):
        while True:

            self.dispatcher.dispatch()
            n_new_users = np.random.poisson(1)

            for i in range(0, n_new_users):
                self.dispatcher.add_new_user(User(env, self.num_users, self.map_size))
                print(f"User {self.num_users} joins the simulation at time {env.now}")
                self.num_users += 1

            yield self.env.timeout(6)  # Dispatch cars every 1 minute


# Setup and run the simulation
env = simpy.Environment()
car_sharing_sim = CarSharingSimulation(env=env, num_cars=5, num_users=10, map_size=(10, 10), simulation_time=2000)
car_sharing_sim.run()
print(f"Customers Served: {n_served_customers}")
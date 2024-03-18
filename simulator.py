import time
from functools import wraps
import random
import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment
import simpy


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
    def __init__(self, env, users, available_cars):
        self.env = env
        self.users = users
        self.available_cars = available_cars
        # add tracker of positions for available cars
        self.user_coordinates = {user.id: user.position for user in self.users}
        self.car_coordinates = {car.id: car.position for car in self.available_cars}

    def solve(self) -> float:
        x_index, y_index = linear_sum_assignment(self.cost_matrix)
        optimal_value = self.cost_matrix[x_index, y_index].sum()

        return optimal_value

    def add_new_user(self, user):
        self.users.append(user)
        self.user_coordinates[user.id] = user.position

    def dispatch(self):
        # Generate all possible combinations of users and cars
        assignments = list(itertools.product(self.users, self.available_cars))
        global n_served_customers

        # Find the assignment that minimizes total distance
        min_distance = float("inf")
        best_assignment = None
        for user, car in assignments:
            distance = car.get_distance_to_user(user)
            if distance < min_distance:
                min_distance = distance
                best_assignment = (user, car)

        # Assign cars to users based on the globally optimal assignment
        if best_assignment:
            user, car = best_assignment
            self.available_cars.remove(car)
            self.env.process(user.use_car(car))
            # self.car_coordinates.pop(car.id, None)
            self.available_cars.append(car)
            car.position = user.position
            # self.car_coordinates[car.id] = car.position
            self.users.remove(user)
            n_served_customers += 1


class CarSharingSimulation:
    def __init__(self, env, num_cars, num_users, map_size, simulation_time):
        self.env = env
        self.num_cars = num_cars
        self.num_users = num_users
        self.map_size = map_size
        self.cars = [Car(env, i, self.map_size, capacity=1) for i in range(num_cars)]
        self.users = [User(env, i, self.map_size) for i in range(num_users)]
        self.available_cars = list(self.cars)  # Initially, all cars are available
        self.dispatcher = Dispatcher(env, self.users, self.available_cars)
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
            n_new_users = np.random.poisson(0.2)

            for i in range(0, n_new_users):
                self.dispatcher.add_new_user(User(env, self.num_users, self.map_size))
                print(f"User {self.num_users} joins the simulation at time {env.now}")
                self.num_users += 1

            yield self.env.timeout(6)  # Dispatch cars every 1 minute


# Setup and run the simulation
env = simpy.Environment()
car_sharing_sim = CarSharingSimulation(env=env, num_cars=1, num_users=2000, map_size=(10, 10), simulation_time=2000)
car_sharing_sim.run()
print(f"Customers Served: {n_served_customers}")
import random

from solvers import MILPSolver
from utils import timeit

import numpy as np
from scipy.optimize import linear_sum_assignment
import simpy
from scipy.stats import norm
from scipy.spatial import distance
import pandas as pd

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
    def __init__(self, env, id, map_size, simulation_id):
        self.env = env
        self.id = id
        self.simulation_id = simulation_id
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

    def save_assignment_cost(self, assignment_cost):
        self.assignment_cost = round(assignment_cost, ndigits=3)

    def use_car(self, car, ride_distance):
        """User action to use a car."""
        print(f"User {self.id} takes car {car.id} at time {self.env.now}")
        global simulation_data
        with car.request() as request:

            yield request
            waiting_time = round(self.env.now - self.request_activation_time, 2)
            service_start_time = round(self.env.now, 2)
            yield self.env.timeout(ride_distance)
            request_completion_time = round(self.env.now, 2)
            service_time = round(request_completion_time - service_start_time, 2)
            order_fullfilment_time = round(request_completion_time - self.request_activation_time, 2)
        
        request_data={"user_id": self.id, "car_id": car.id, "activation_time": self.request_activation_time,
              "request_completion_time": request_completion_time,"waiting_time": waiting_time,
              "service_time": service_time, "order_fullfilment_time": order_fullfilment_time, 
              "assignment_cost": self.assignment_cost, "simulation_id": self.simulation_id}
        simulation_data.append(request_data)

        # User returns the car
        print(f"User {self.id} leaves car {car.id} at time {self.env.now}")


class Dispatcher:
    """Dispatcher class to manage car-user assignments."""
    def __init__(self, env, users, available_cars, solver):
        self.env = env
        self.solver = solver
        self.users = {user.id: user for user in users}
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

    def remove_served_user(self, user):
        """Remove user from the simulation."""
        self.users.pop(user.id, None)

    def set_car_to_busy(self, car):
        """Remove car from set of availabile cars."""
        self.available_cars.pop(car.id, None)

    def set_car_to_free(self, car):
        """Add car to set of available cars."""
        self.available_cars.update({car.id: car})

    def move_car_to_ride_destination(self, car, user):
        """Set car to position of customer's destination."""
        car.coordinates_start = user.coordinates_start

    def dispatch(self):
        """Dispatch cars to users."""
        global n_served_customers

        if not self.available_cars or not self.users:
            return
        
        optimal_solution = self.solve(self.solver)
        # Assign cars to users based on the globally optimal assignment
        for optimal_match in optimal_solution:
            #improve readability, split into different methods to be called within the for loop
            matched_car_id = optimal_match.get("cars")
            matched_user_id = optimal_match.get("users")
            assignment_cost = optimal_match.get("assignment_cost")
            car = self.available_cars.get(matched_car_id)
            user = self.users.get(matched_user_id)
            user.save_assignment_cost(assignment_cost)
            self.global_assignment_cost+=assignment_cost

            self.set_car_to_busy(car)
            self.env.process(user.use_car(car, assignment_cost))
            self.move_car_to_ride_destination(car, user)
            self.n_served_customers += 1
            self.remove_served_user(user)
            self.set_car_to_free(car)


class RideSharingSimulation:
    """Simulation of car sharing system."""
    def __init__(self, env, id, num_cars, num_users, map_size, simulation_time, inflow_rate):
        self.env = env
        self.id = id
        self.num_cars = num_cars
        self.num_users = num_users
        self.map_size = map_size
        self.cars = [Car(env, i, self.map_size, capacity=1) for i in range(num_cars)]
        self.users = [User(env, i, self.map_size, self.id) for i in range(num_users)]
        self.available_cars = self.cars  # Initially, all cars are available
        self.dispatcher = Dispatcher(env, self.users, self.cars, MILPSolver(decision_variable_type="real", solver_name="cbc"))
        self.simulation_time = simulation_time
        self.expected_customer_inflow = inflow_rate

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
            n_new_users = np.random.poisson(self.expected_customer_inflow)

            for _ in range(0, n_new_users):
                self.dispatcher.add_new_user(User(env, self.num_users, self.map_size, self.id))
                print(f"User {self.num_users} joins the simulation at time {env.now}")
                self.num_users += 1

            yield self.env.timeout(15)  # Dispatch cars every 1 minute

    def get_global_cost(self):
        """Get simulation data for analysis."""
        return round(self.dispatcher.global_assignment_cost/self.dispatcher.n_served_customers, ndigits=2)


# Setup and run the simulation

simulation_results = {"simulation_id": [], "global_cost": []}
    
for simulation_id in range(0, 10): 
    env = simpy.Environment()
    simulation = RideSharingSimulation(env=env, id=simulation_id, num_cars=20, num_users=40, map_size=(10, 10), simulation_time=200, inflow_rate=10)
    simulation.run()
    simulation_results.get("simulation_id").append(simulation_id)
    simulation_results.get("global_cost").append(simulation.get_global_cost())
 
simulation_data = pd.DataFrame(simulation_data).to_csv("sim_data.csv", index=False)
simulation_results = pd.DataFrame(simulation_results).to_csv("sim_results.csv", index=False)

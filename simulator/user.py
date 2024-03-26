import numpy as np

import logging

class User:
    """Representation of a user in the simulation."""
    def __init__(self, env, id, map_size, coordinate_generator):
        self.env = env
        self.id = id
        self.map_size = map_size
        self.coordinates_start = coordinate_generator.sample()
        self.coordinates_destination = coordinate_generator.sample()
        self.served_status = False
        self.service_start_time = None
        self.request_completion_time = None
        self.request_activation_time = self.env.now
        self.assignment_cost = np.nan
        self.waiting_time = np.nan
        self.service_time = np.nan
        self.order_fullfilment_time = np.nan

    def to_dict(self):
        return vars(self)

    def use_car(self, car):
        """User action to use a car."""
        with car.request() as request:

            yield request

            logging.debug(f"User {self.id} takes car {car.id} at time {self.env.now}")

            #note that start to customer and customer to destination are not split in this logic
            self.waiting_time = round(self.env.now - self.request_activation_time, 2)
            self.service_start_time = round(self.env.now, 2)
            yield self.env.timeout(self.assignment_cost)

            logging.debug(f"User {self.id} exits car {car.id} at time {self.env.now}")

            self.request_completion_time = round(self.env.now, 2)
            self.service_time = round(self.request_completion_time - self.service_start_time, 2)
            self.order_fullfilment_time = round(self.request_completion_time - self.request_activation_time, 2)
            self.served_status = True
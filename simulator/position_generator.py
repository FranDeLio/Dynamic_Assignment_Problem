from abc import ABC, abstractmethod
import random

class AbstractCoordinateGenerator(ABC):

    @abstractmethod
    def sample(self):
        pass

class UniformCoordinateGenerator(AbstractCoordinateGenerator):

    def __init__(self, max_x_axis, max_y_axis):
        self.max_x_axis=max_x_axis
        self.max_y_axis=max_y_axis

    def sample(self):
        x_coordinate = random.uniform(0, self.max_x_axis)
        y_coordinate = random.uniform(0, self.max_y_axis)

        return (x_coordinate, y_coordinate)
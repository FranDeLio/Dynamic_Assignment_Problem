import simpy

class Car(simpy.Resource):
    """Representation of a car in the simulation."""
    def __init__(self, env, id, map_size, coordinate_generator, capacity):
        self.id = id
        self.map_size = map_size
        self.coordinates_start = coordinate_generator.sample()
        super().__init__(env, capacity)
from abc import ABC, abstractmethod
from scipy.stats import norm
from scipy.spatial import distance
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
#from ortools.graph.python import min_cost_flow
import pyomo.environ as pe
import pyomo.opt as po
import logging


logging.getLogger("pyomo.core").setLevel(logging.ERROR)


class OptimalTransportProblem(ABC):
    def __init__(
        self, 
    ) -> None:

        x=1

    @abstractmethod
    def get_optimal_solution(self) -> float:
        pass


class MILPSolver(OptimalTransportProblem):
    def __init__(
        self
    ) -> None:
        #super().__init__(available_cars, users)

        x=1

    def get_optimal_solution(self, available_cars, users,
        decision_variable_type: str = "real", solver_name: str = "cbc"
    ) -> float:
        
        if decision_variable_type == "real":
            decision_domain = pe.NonNegativeReals
        else:
            decision_domain = pe.Binary

        self.source_coordinates = [car.position for car in available_cars.values()]
        self.destination_coordinates = [user.position for user in users.values()]
        self.cost_matrix = distance.cdist(
            self.source_coordinates, self.destination_coordinates, "euclidean"
        )
        

        self.available_cars_ids = [car.id for car in available_cars.values()]
        self.users_ids = [user.id for user in users.values()]
        #self.cost_matrix = np.vstack([self.users_ids, self.cost_matrix])
        import itertools
        #self.cost_matrix = np.stack([list(itertools.chain.from_iterable([[[0]],[self.available_cars_ids]])), self.cost_matrix])

        solver = po.SolverFactory(solver_name)
        model = pe.ConcreteModel()

        model.source = pe.Set(initialize=self.available_cars_ids)
        model.destination = pe.Set(initialize=self.users_ids)

        df=pd.DataFrame(self.cost_matrix)
        df.insert(loc=0, column='insert', value=self.available_cars_ids)
        df=df.set_index('insert')
        df.columns=self.users_ids
        cost = (
            df
            .stack()
            .to_dict()
        )

        model.cost = pe.Param(model.source, model.destination, initialize=cost)
        model.y = pe.Var(
            model.source, model.destination, domain=decision_domain
        )  # variable: 1 if assign parameters set k to city c else 0.

        expression = sum(
            - model.cost[c, k] * model.y[c, k]
            for c in model.source
            for k in model.destination
        )
        model.obj = pe.Objective(sense=pe.minimize, expr=expression)

        def serve_all_destinations(model, k):
            # assign exactly one origin set c to every destination k.
            constraint = sum(model.y[c, k] for c in model.source) <= 1
            return constraint

        def origin_unicity(model, c):
            # an origin c can only be assign to one given destination k.
            constraint = sum(model.y[c, k] for k in model.destination) <= 1
            return constraint

        model.serve_all_destinations = pe.Constraint(
            model.destination, rule=serve_all_destinations
        )

        model.origin_unicity = pe.Constraint(model.source, rule=origin_unicity)

        result = solver.solve(model, timelimit=60)

        solution_df = (pd.Series(model.y.extract_values())
                   .reset_index()
                   .rename(columns={'level_0': 'users', 'level_1': 'cars', 0: 'y'}))
        #print(solution_df)
        solution_df = solution_df.loc[solution_df.y == 1, ["users", "cars"]].set_index("cars")

        #print(solution_df)

        result = solution_df.to_dict()["users"]

        #print(result)


        
        return result

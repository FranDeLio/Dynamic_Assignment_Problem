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


class AssignmentSolver(ABC):
    #this is a good place to specify the objective function to be used

    @abstractmethod
    def get_optimal_solution(self) -> float:
        pass


class MILPSolver(AssignmentSolver):
    def __init__(
        self, decision_variable_type: str = "real", solver_name: str = "cbc"
    ) -> None:
        
        self.decision_variable_type = decision_variable_type
        self.solver_name = solver_name

    def get_optimal_solution(self, available_cars, users,
    ) -> float:
        
        if self.decision_variable_type == "real":
            decision_domain = pe.NonNegativeReals
        else:
            decision_domain = pe.Binary

        self.source_coordinates = [car.coordinates_start for car in available_cars.values()]
        self.destination_coordinates = [user.coordinates_start for user in users.values()]
        self.cost_matrix = distance.cdist(
            self.source_coordinates, self.destination_coordinates, "euclidean"
        )

        self.available_cars_ids = [car.id for car in available_cars.values()]
        self.users_ids = [user.id for user in users.values()]

        solver = po.SolverFactory(self.solver_name)
        model = pe.ConcreteModel()

        model.source = pe.Set(initialize=self.available_cars_ids)
        model.destination = pe.Set(initialize=self.users_ids)

        cost = (pd.DataFrame(self.cost_matrix, index=self.available_cars_ids, columns=self.users_ids)
                .stack()
                .to_dict())
        
        print(cost)

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
                   .rename(columns={'level_0': 'cars', 'level_1': 'users', 0: 'y'})
                   .loc[lambda x: x.y == 1, ["users", "cars"]]
                   .reset_index(drop=True))
        
        n_rows = solution_df.shape[0]

        for i in range(0, n_rows):
           solution_df.loc[i,"assignment_cost"] = cost[solution_df.loc[i,"cars"], solution_df.loc[i,"users"]]
        
        result = solution_df.to_dict(orient='records')
        
        return result

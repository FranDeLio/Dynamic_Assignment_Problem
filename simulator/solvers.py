from scipy.spatial import distance
import pandas as pd
import pyomo.environ as pe
import pyomo.opt as po

from abc import ABC, abstractmethod
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

        self.car_coordinates = [car.coordinates_start for car in available_cars.values()]
        self.destination_coordinates = [user.coordinates_start for user in users.values()]
        self.cost_matrix = distance.cdist(
            self.car_coordinates, self.destination_coordinates, "euclidean"
        )

        self.available_cars_ids = [car.id for car in available_cars.values()]
        self.users_ids = [user.id for user in users.values()]
        self.n_assignments = min(len(self.available_cars_ids), len(self.users_ids))

        solver = po.SolverFactory(self.solver_name)
        model = pe.ConcreteModel()

        model.cars = pe.Set(initialize=self.available_cars_ids)
        model.users = pe.Set(initialize=self.users_ids)

        assignment_cost = (pd.DataFrame(self.cost_matrix, index=self.available_cars_ids, columns=self.users_ids)
                .stack()
                .to_dict())
        
        model.assignment_cost = pe.Param(model.cars, model.users, initialize=assignment_cost)
        model.y = pe.Var(
            model.cars, model.users, domain=decision_domain
        )  # variable: 1 if assign parameters set k to city c else 0.

        expression = sum(
            model.assignment_cost[c, k] * model.y[c, k]
            for c in model.cars
            for k in model.users
        )
        model.obj = pe.Objective(sense=pe.minimize, expr=expression)

        def each_car_serves_one_user_max(model, k):
            # assign exactly one origin set c to every destination k.
            constraint = sum(model.y[c, k] for c in model.cars) <= 1
            return constraint

        def each_user_is_served_by_one_car_max(model, c):
            # an origin c can only be assign to one given destination k.
            constraint = sum(model.y[c, k] for k in model.users) <= 1
            return constraint
        
        def maximal_injective_assignment(model):
            # a minimum of min(|origin|,|destination|) matches ought to be made.
            constraint = sum(model.y[c, k] for k in model.users for c in model.cars) == self.n_assignments
            return constraint

        model.each_car_serves_one_user_max = pe.Constraint(model.users, rule=each_car_serves_one_user_max)
        model.each_user_is_served_by_one_car_max = pe.Constraint(model.cars, rule=each_user_is_served_by_one_car_max)
        model.maximal_injective_assignment = pe.Constraint(rule=maximal_injective_assignment)

        result = solver.solve(model, timelimit=60)

        solution_df = (pd.Series(model.y.extract_values())
                   .reset_index()
                   .rename(columns={'level_0': 'cars', 'level_1': 'users', 0: 'y'})
                   .loc[lambda x: x.y == 1, ["users", "cars"]]
                   .reset_index(drop=True))
        
        n_rows = solution_df.shape[0]

        for i in range(0, n_rows):
           solution_df.loc[i,"assignment_cost"] = assignment_cost[solution_df.loc[i,"cars"], solution_df.loc[i,"users"]]
        
        result = solution_df.to_dict(orient='records')
        
        return result

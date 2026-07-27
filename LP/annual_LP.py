import pyomo.environ as pyo

# abstract model formulation
model = pyo.AbstractModel()

# sets
model.TECHNOLOGIES = pyo.Set()
model.SCENARIOS = pyo.Set()

# parameters
model.demand = pyo.Param()
model.unit_cost = pyo.Param(model.TECHNOLOGIES, model.SCENARIOS)
model.previous_gen = pyo.Param(model.TECHNOLOGIES)

# variables
model.gen = pyo.Var(model.TECHNOLOGIES, domain=pyo.NonNegativeReals)

# objective
def obj_fun(model):
    return sum(
        model.unit_cost[i, j] * model.gen[i]
        for i in model.TECHNOLOGIES 
        for j in model.SCENARIOS
    ) / len(model.SCENARIOS)
model.obj = pyo.Objective(rule=obj_fun)

# demand constraints
def demand_satisfaction(model):
    return sum(model.gen[i] for i in model.TECHNOLOGIES) == model.demand

model.demand_constraints = pyo.Constraint(
    rule=demand_satisfaction
)

# growth rate constraints
def growth_constraints_lower(model, i):
    return model.gen[i] >= model.previous_gen[i] * 0.7    
def growth_constraints_upper(model, i):
    return model.gen[i] <= model.previous_gen[i] * 1.3

model.growth_constraints = pyo.Constraint(
    model.TECHNOLOGIES,
    rule=growth_constraints_lower
)

model.growth_constraints_2 = pyo.Constraint(
    model.TECHNOLOGIES,
    rule=growth_constraints_upper
)


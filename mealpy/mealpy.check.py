from mealpy import GWO, FloatVar

problem = {
    "obj_func": lambda x: sum(x**2),
    "bounds": FloatVar(lb=[-5]*3, ub=[5]*3),
    "minmax": "min",
    "log_to": None
}

model = GWO.OriginalGWO(epoch=10, pop_size=10)
model.solve(problem)

# g_best içine bak
print("g_best:", model.g_best)
print("g_best tipi:", type(model.g_best))
print("g_best attributes:", [a for a in dir(model.g_best) if not a.startswith('__')])

# g_best'in içindekiler
print("\ng_best.solution:", model.g_best.solution)
print("g_best.target:", model.g_best.target)
print("g_best.target tipi:", type(model.g_best.target))
print("g_best.target attributes:", [a for a in dir(model.g_best.target) if not a.startswith('__')])
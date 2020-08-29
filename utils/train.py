

# tasks
classifications = ["sede_icdo3", "morfologia_icdo3"]
regressions = ["stadio", "dimensioni"]

# model
model = "models/xgdtree.py"
hyperparams = {"num_trees": 10}

# operations (again)
# model.train
# model.save
# model.evaluate
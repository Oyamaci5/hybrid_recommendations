import pickle
with open("comparison_algorithms/results/ha_raw_results.pkl", "rb") as f:
    data = pickle.load(f)
print(data)
print(type(data))
print(data.keys())
print(data["ha_raw_results"].keys())
print(data["ha_raw_results"]["F1"].keys())
print(data["ha_raw_results"]["F1"]["results"].keys())
print(data["ha_raw_results"]["F1"]["results"]["F1"].keys())
print(data["ha_raw_results"]["F1"]["results"]["F1"]["results"].keys())
print(data["ha_raw_results"]["F1"]["results"]["F1"]["results"]["F1"].keys())
print(data["ha_raw_results"]["F1"]["results"]["F1"]["results"]["F1"]["results"].keys())
print(data["ha_raw_results"]["F1"]["results"]["F1"]["results"]["F1"]["results"].keys())
import pickle
import torch

with open("../model_output.pickle", "rb") as file:
	output = pickle.load(file)
	
print(output.shape)
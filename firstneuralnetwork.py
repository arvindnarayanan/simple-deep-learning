import torch
import torch.nn as nn
import torch.optim as optim

#Input Data
distances = torch.tensor([[1.0],[2.0],[3.0],[4.0]], dtype=torch.float32)
times = torch.tensor([[6.96],[12.22],[16.77],[22.21]], dtype=torch.float32)

#Define the Model
model = nn.Sequential(nn.Linear(1,1))

#Defining Loss Function and Optimizer
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


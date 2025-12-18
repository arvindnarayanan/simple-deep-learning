import torch

torch.manual_seed(1)
random = torch.rand(5,1)
print(random)

distance = torch.tensor(25.0)
print(distance.shape)
distance = distance.unsqueeze(0)
print(distance.shape)


distance = torch.tensor([25.0])
print(distance.shape)
distance = torch.tensor([[25.0]])
print(distance.shape)
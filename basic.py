import torch
print("PyTorch version: " + torch.__version__)
print("CUDA available: ", torch.cuda.is_available())


if torch.cuda.is_available():
    print("GPU name: "+ torch.cuda.get_device_name(0))
else:
    print("No GPU detected")

x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
print(x.grad)

w = torch.tensor(0.0,requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

x = torch.tensor(1.0)
y_true = torch.tensor(3.0)

y_pred = w*x + b
error = (y_pred - y_true)**2

error.backward()
print(w.grad, b.grad)
print(w)
import torch 
torch.manual_seed(42)

random_tensor_1 = torch.rand(3, 3)

print("First execution: ")
print(random_tensor_1)

torch.manual_seed(42)

random_tensor_2 = torch.rand(3,3)

print("Second execution: ")
print(random_tensor_2)

print("Are they equal: ")
print(torch.equal(random_tensor_1, random_tensor_2))
tensor = torch.tensor([1,2,3])
print(tensor.device)
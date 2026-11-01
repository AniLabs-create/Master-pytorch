import torch

zeros = torch.zeros(size=(2,3))

ones = torch.ones(size=(3,3))

random_tensors = torch.rand(size=(2,4))

normal_random_tensors = torch.randn(size=(3,3))

nums = torch.arange(5,20,1)

print(zeros)

print(ones)

print(random_tensors)

print(normal_random_tensors)

print(nums)

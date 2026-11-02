import torch 
number = torch.tensor([45.0, 78.0, 92.0, 34.0, 88.0])
min = torch.min(number)
max = torch.max(number)
mean = torch.mean(number)
sum = torch.sum(number)
min_index =  torch.argmin(number)
max_index = torch.argmax(number)
print(f"Minimum Number: {min}")
print(f"Index of the minimum value is : {min_index}")

print(f"Maximum Number: {max}")
print(f"Index of the maximum value is : {max_index}")

print(f"The Average of all the give numbers is : {mean}")

print(f"The Sum of all the numbers in the list is: {sum}")
import torch

a = torch.tensor([[[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]]])
# filter = torch.tensor([[True, False, True], [True, True, False]])
# b = torch.tensor([[[10,20,30], [40,50,60], [70,80,90]], [[10,20,30], [41,52,63], [71,81,91]]])

# a[0][filter[0]] = b[0][filter[0]]
# a[1][filter[1]] = b[1][filter[1]]

# expected tensor
# a = torch.tensor([[[10,20,30], [4,5,6], [70,80,90]], [[10,20,30], [41,52,63], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]]])
print(a)
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")

    #case 1:
    x = torch.ones(5, device= device)

    #case 2:
    y = torch.ones(5)
    y = y.to(device)
